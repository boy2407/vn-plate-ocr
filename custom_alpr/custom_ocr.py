"""A small project-owned CNN + Transformer + CTC OCR baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ"
BLANK_INDEX = 0
CHAR_TO_INDEX = {character: index + 1 for index, character in enumerate(ALPHABET)}
INDEX_TO_CHAR = {index: character for character, index in CHAR_TO_INDEX.items()}
NUM_CLASSES = len(ALPHABET) + 1
PAD_INDEX = 0
BOS_INDEX = 1
EOS_INDEX = 2
DECODER_CHAR_TO_INDEX = {character: index + 3 for index, character in enumerate(ALPHABET)}
DECODER_INDEX_TO_CHAR = {index: character for character, index in DECODER_CHAR_TO_INDEX.items()}
DECODER_VOCAB_SIZE = len(ALPHABET) + 3


class BlurPool2d(nn.Module):
    """Fixed low-pass filter followed by subsampling to reduce aliasing."""

    def __init__(self, channels: int, stride: tuple[int, int]):
        super().__init__()
        self.channels = channels
        self.stride = stride
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = torch.outer(kernel, kernel)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None])

    def forward(self, features: Tensor) -> Tensor:
        kernel = self.kernel.to(dtype=features.dtype).expand(self.channels, 1, 3, 3)
        return nn.functional.conv2d(
            features, kernel, stride=self.stride, padding=1, groups=self.channels
        )


class ResidualBlurBlock(nn.Module):
    """Residual feature block with blur applied before spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.main_blur = BlurPool2d(out_channels, stride)
        self.skip_blur = BlurPool2d(out_channels, stride)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        main = self.main_blur(self.main(features))
        skip = self.skip_blur(self.skip(features))
        return self.activation(main + skip)


class PlateOCR(nn.Module):
    """Recognize a plate crop with CNN features, Transformer, and CTC logits.

    Input is a normalized RGB batch shaped ``[batch, 3, 64, 128]``. Output logits
    are ``[time, batch, classes]`` for ``torch.nn.CTCLoss``; class 0 is CTC blank.
    """

    def __init__(
        self,
        image_height: int = 64,
        image_width: int = 128,
        model_dim: int = 192,
        attention_heads: int = 4,
        encoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_height % 16 or image_width % 4:
            raise ValueError("image_height must divide by 16 and image_width by 4")

        self.image_height = image_height
        self.image_width = image_width
        self.time_steps = image_width // 4
        self.cnn = nn.Sequential(
            ResidualBlurBlock(3, 64, (2, 2)),
            ResidualBlurBlock(64, 128, (2, 2)),
            ResidualBlurBlock(128, 192, (2, 1)),
            ResidualBlurBlock(192, 256, (2, 1)),
        )
        cnn_height = image_height // 16
        self.projection = nn.Linear(256 * cnn_height, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.time_steps, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, NUM_CLASSES)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw CTC logits shaped ``[time, batch, classes]``."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB input shaped [batch, 3, height, width]")
        if tuple(images.shape[-2:]) != (self.image_height, self.image_width):
            raise ValueError(
                f"Expected image size {(self.image_height, self.image_width)}, "
                f"got {tuple(images.shape[-2:])}"
            )
        features = self.cnn(images)  # [B, C=256, H=4, W=32]
        sequence = features.permute(0, 3, 1, 2).flatten(2)  # [B, W=32, C*H]
        sequence = self.projection(sequence) + self.position_embedding
        sequence = self.encoder(sequence)
        logits = self.classifier(self.output_norm(sequence))
        return logits.transpose(0, 1)  # [T, B, classes]

    def input_lengths(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Return each image's output sequence length for CTC loss."""
        return torch.full((batch_size,), self.time_steps, dtype=torch.long, device=device)


class PlateOCR2D(nn.Module):
    """Layout-free 2D visual Transformer encoder with an autoregressive decoder.

    The encoder sees every cell of the CNN feature map with row/column positions.
    The decoder learns the reading order from full-plate transcriptions; no
    single-row/double-row labels or character boxes are required.
    """

    def __init__(
        self,
        image_height: int = 64,
        image_width: int = 128,
        model_dim: int = 192,
        attention_heads: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        max_output_length: int = 14,
    ):
        super().__init__()
        if image_height % 16 or image_width % 4:
            raise ValueError("image_height must divide by 16 and image_width by 4")
        if max_output_length < 2:
            raise ValueError("max_output_length must allow at least BOS and EOS")

        self.image_height = image_height
        self.image_width = image_width
        self.feature_height = image_height // 16
        self.feature_width = image_width // 4
        self.max_output_length = max_output_length
        self.cnn = nn.Sequential(
            ResidualBlurBlock(3, 64, (2, 2)),
            ResidualBlurBlock(64, 128, (2, 2)),
            ResidualBlurBlock(128, 192, (2, 1)),
            ResidualBlurBlock(192, 256, (2, 1)),
        )
        self.visual_projection = nn.Linear(256, model_dim)
        self.row_embedding = nn.Parameter(torch.zeros(1, self.feature_height, 1, model_dim))
        self.column_embedding = nn.Parameter(torch.zeros(1, 1, self.feature_width, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.encoder_norm = nn.LayerNorm(model_dim)

        self.token_embedding = nn.Embedding(DECODER_VOCAB_SIZE, model_dim, padding_idx=PAD_INDEX)
        self.decoder_position = nn.Parameter(torch.zeros(1, max_output_length, model_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, DECODER_VOCAB_SIZE)
        nn.init.trunc_normal_(self.row_embedding, std=0.02)
        nn.init.trunc_normal_(self.column_embedding, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)

    def encode(self, images: Tensor) -> Tensor:
        """Encode crops to visual memory shaped ``[batch, rows*cols, model_dim]``."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB input shaped [batch, 3, height, width]")
        if tuple(images.shape[-2:]) != (self.image_height, self.image_width):
            raise ValueError(
                f"Expected image size {(self.image_height, self.image_width)}, "
                f"got {tuple(images.shape[-2:])}"
            )
        features = self.cnn(images)  # [B, 256, H=4, W=32]
        visual = features.permute(0, 2, 3, 1)  # [B, H=4, W=32, 256]
        visual = self.visual_projection(visual)
        visual = visual + self.row_embedding + self.column_embedding
        batch_size = visual.shape[0]
        sequence = visual.reshape(batch_size, self.feature_height * self.feature_width, -1)
        return self.encoder_norm(self.encoder(sequence))

    def decode(self, memory: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Return next-token logits for teacher-forced decoder input IDs."""
        length = decoder_input_ids.shape[1]
        if length > self.max_output_length:
            raise ValueError(f"Decoder input exceeds {self.max_output_length} tokens")
        positions = self.decoder_position[:, :length]
        target = self.token_embedding(decoder_input_ids) + positions
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=decoder_input_ids.device),
            diagonal=1,
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(PAD_INDEX),
        )
        return self.classifier(self.decoder_norm(decoded))

    def forward(self, images: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Return teacher-forced logits shaped ``[batch, steps, vocabulary]``."""
        return self.decode(self.encode(images), decoder_input_ids)

    @torch.inference_mode()
    def generate(self, images: Tensor, max_output_length: int | None = None) -> list[str]:
        """Greedily generate plate strings from complete plate crops."""
        max_length = max_output_length or self.max_output_length
        if max_length > self.max_output_length:
            raise ValueError(f"max_output_length cannot exceed {self.max_output_length}")
        memory = self.encode(images)
        batch_size = images.shape[0]
        generated = torch.full((batch_size, 1), BOS_INDEX, dtype=torch.long, device=images.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
        for _ in range(max_length - 1):
            next_logits = self.decode(memory, generated)[:, -1]
            next_ids = next_logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, PAD_INDEX), next_ids)
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            finished |= next_ids.eq(EOS_INDEX)
            if finished.all():
                break
        return decode_targets(generated)


def normalize_text(text: str) -> str:
    """Remove plate-label separators while preserving meaningful characters."""
    return "".join(
        character
        for character in text.upper()
        if not character.isspace() and character not in {"-", "."}
    )


def encode_text(text: str) -> list[int]:
    """Map a plate transcription to nonblank class indices."""
    normalized = normalize_text(text)
    if not normalized or any(character not in CHAR_TO_INDEX for character in normalized):
        raise ValueError("Text must contain 1 or more uppercase A-Z/Đ/0-9 characters")
    return [CHAR_TO_INDEX[character] for character in normalized]


def encode_target(text: str) -> list[int]:
    """Encode one complete transcription for the autoregressive decoder."""
    normalized = normalize_text(text)
    if not normalized or any(character not in DECODER_CHAR_TO_INDEX for character in normalized):
        raise ValueError("Text must contain 1 or more uppercase A-Z/Đ/0-9 characters")
    return [BOS_INDEX, *(DECODER_CHAR_TO_INDEX[character] for character in normalized), EOS_INDEX]


def sequence_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Teacher-forced cross-entropy; target IDs include BOS, text, EOS, and PAD."""
    expected = target_ids[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), expected.reshape(-1), ignore_index=PAD_INDEX
    )


def decode_targets(target_ids: Tensor) -> list[str]:
    """Convert decoder token IDs to strings, stopping at EOS and ignoring PAD/BOS."""
    decoded = []
    for sequence in target_ids.detach().cpu().tolist():
        characters = []
        for index in sequence:
            if index == EOS_INDEX:
                break
            if index in DECODER_INDEX_TO_CHAR:
                characters.append(DECODER_INDEX_TO_CHAR[index])
        decoded.append("".join(characters))
    return decoded


def ctc_loss(logits: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
    """Calculate CTC loss from model logits and concatenated encoded targets."""
    time_steps, batch_size = logits.shape[:2]
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=logits.device)
    return nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)(
        logits.log_softmax(dim=-1), targets, input_lengths, target_lengths
    )


def greedy_decode(logits: Tensor) -> list[str]:
    """Decode CTC logits with argmax, blank removal, and repeat collapse."""
    if logits.ndim != 3:
        raise ValueError("Expected logits shaped [time, batch, classes]")
    paths = logits.argmax(dim=-1).transpose(0, 1).detach().cpu().tolist()
    decoded = []
    for path in paths:
        characters = []
        previous = None
        for index in path:
            if index not in (BLANK_INDEX, previous):
                characters.append(INDEX_TO_CHAR[index])
            previous = index
        decoded.append("".join(characters))
    return decoded
