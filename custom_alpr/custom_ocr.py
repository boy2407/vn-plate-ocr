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


class PermutationDecoderLayer(nn.Module):
    """Pre-norm layer: position queries attend to context tokens, then to the image."""

    def __init__(self, model_dim: int, attention_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.context_attention = nn.MultiheadAttention(
            model_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(model_dim)
        self.memory_attention = nn.MultiheadAttention(
            model_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        memory: Tensor,
        attention_mask: Tensor,
        context_padding_mask: Tensor,
    ) -> Tensor:
        context = self.context_norm(context)
        attended, _ = self.context_attention(
            self.query_norm(queries),
            context,
            context,
            attn_mask=attention_mask,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        normalized = self.memory_norm(queries)
        attended, _ = self.memory_attention(normalized, memory, memory, need_weights=False)
        queries = queries + self.dropout(attended)
        return queries + self.dropout(self.feedforward(self.feedforward_norm(queries)))


class PlateOCRParseq(PlateOCR2D):
    """PlateOCR2D encoder with a PARSeq-style permutation language-model decoder.

    Each output position is a learned position query; characters appear only as
    context keys/values. Training averages the loss over several reading orders,
    each expressed as an attention mask, so the decoder cannot rely on the
    previous character alone. Inference decodes left to right, then refines every
    position once while it sees all other predicted characters (cloze mask).
    """

    def __init__(
        self,
        *args,
        permutations: int = 6,
        refine_iterations: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if permutations < 1 or permutations % 2:
            raise ValueError("permutations must be a positive even number (orders + flips)")
        self.permutations = permutations
        self.refine_iterations = refine_iterations
        model_dim = self.token_embedding.embedding_dim
        attention_heads = self.decoder.layers[0].self_attn.num_heads
        dropout = self.decoder.layers[0].dropout.p
        layer_count = len(self.decoder.layers)
        del self.decoder
        self.position_queries = nn.Parameter(torch.zeros(1, self.max_output_length, model_dim))
        self.decoder_layers = nn.ModuleList(
            PermutationDecoderLayer(model_dim, attention_heads, dropout) for _ in range(layer_count)
        )
        nn.init.trunc_normal_(self.position_queries, std=0.02)

    def decode_positions(
        self, memory: Tensor, context_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Logits for every position query given context IDs ``[BOS, t0, t1, ...]``."""
        length = context_ids.shape[1]
        if length > self.max_output_length:
            raise ValueError(f"Decoder input exceeds {self.max_output_length} tokens")
        context = self.token_embedding(context_ids) + self.decoder_position[:, :length]
        queries = self.position_queries[:, :length].expand(context_ids.shape[0], -1, -1)
        padding = context_ids.eq(PAD_INDEX)
        for layer in self.decoder_layers:
            queries = layer(queries, context, memory, attention_mask, padding)
        return self.classifier(self.decoder_norm(queries))

    def decode(self, memory: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Left-to-right (causal) logits, matching the PlateOCR2D teacher-forced API."""
        mask = permutation_attention_mask(
            torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
        )
        return self.decode_positions(memory, decoder_input_ids, mask)

    def permutation_loss(
        self,
        images: Tensor,
        decoder_input_ids: Tensor,
        target_ids: Tensor,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Mean cross-entropy over left-to-right, right-to-left, and random orders."""
        memory = self.encode(images)
        length = decoder_input_ids.shape[1]
        device = decoder_input_ids.device
        orders = [torch.arange(length, device=device)]
        while len(orders) < self.permutations // 2:
            orders.append(torch.randperm(length, device=device))
        orders += [order.flip(0) for order in orders]
        losses = [
            sequence_loss(
                self.decode_positions(memory, decoder_input_ids, permutation_attention_mask(o)),
                target_ids,
                label_smoothing,
            )
            for o in orders
        ]
        return torch.stack(losses).mean()

    @torch.inference_mode()
    def generate_with_confidence(
        self, images: Tensor, max_output_length: int | None = None
    ) -> tuple[list[str], list[float]]:
        """Decode left to right, refine with a cloze mask, and score each plate.

        Confidence is the geometric mean probability of the final character tokens
        (EOS excluded), the same definition used for PlateOCR2D.
        """
        max_length = max_output_length or self.max_output_length
        if max_length > self.max_output_length:
            raise ValueError(f"max_output_length cannot exceed {self.max_output_length}")
        steps = max_length - 1
        memory = self.encode(images)
        batch_size, device = images.shape[0], images.device
        context = torch.full((batch_size, steps), PAD_INDEX, dtype=torch.long, device=device)
        context[:, 0] = BOS_INDEX
        causal = permutation_attention_mask(torch.arange(steps, device=device))
        logits = torch.zeros(batch_size, steps, self.classifier.out_features, device=device)
        for step in range(steps):
            logits[:, step] = self.decode_positions(memory, context, causal)[:, step]
            if step + 1 < steps:
                context[:, step + 1] = logits[:, step].argmax(dim=-1)
            if logits[:, : step + 1].argmax(dim=-1).eq(EOS_INDEX).any(dim=1).all():
                break
        cloze = cloze_attention_mask(steps, device)
        for _ in range(self.refine_iterations):
            context = context_from_prediction(logits.argmax(dim=-1))
            logits = self.decode_positions(memory, context, cloze)
        probabilities, predicted = logits.softmax(dim=-1).max(dim=-1)
        ended = predicted.eq(EOS_INDEX).cumsum(dim=1) > 0
        characters = (~ended) & (predicted >= 3)
        log_sum = torch.where(characters, probabilities.clamp_min(1e-12).log(), 0.0).sum(dim=1)
        counts = characters.sum(dim=1)
        scores = torch.where(counts > 0, (log_sum / counts.clamp_min(1)).exp(), 0.0)
        return decode_targets(predicted), scores.detach().cpu().tolist()

    def generate(self, images: Tensor, max_output_length: int | None = None) -> list[str]:
        return self.generate_with_confidence(images, max_output_length)[0]


def context_from_prediction(predicted: Tensor) -> Tensor:
    """Build ``[BOS, t0, t1, ...]`` context, padding every slot after the first EOS."""
    ended = predicted.eq(EOS_INDEX).cumsum(dim=1) > 0
    context = torch.cat((torch.full_like(predicted[:, :1], BOS_INDEX), predicted[:, :-1]), dim=1)
    after_eos = torch.cat((torch.zeros_like(ended[:, :2]), ended[:, :-2]), dim=1)
    return context.masked_fill(after_eos[:, : context.shape[1]], PAD_INDEX)


def permutation_attention_mask(order: Tensor) -> Tensor:
    """Boolean mask (True = blocked) for position queries over ``[BOS, t0, t1, ...]``.

    Query ``i`` predicts token ``t_i``. It always sees BOS (slot 0) and sees slot
    ``k + 1`` (token ``t_k``) only when ``t_k`` comes before ``t_i`` in ``order``.
    """
    length = order.shape[0]
    rank = torch.empty_like(order)
    rank[order] = torch.arange(length, device=order.device)
    visible = torch.zeros(length, length, dtype=torch.bool, device=order.device)
    visible[:, 0] = True
    visible[:, 1:] = rank[None, : length - 1] < rank[:, None]
    return ~visible


def cloze_attention_mask(length: int, device: torch.device) -> Tensor:
    """Each query sees BOS and every other token, but not its own (slot ``i + 1``)."""
    visible = torch.ones(length, length, dtype=torch.bool, device=device)
    index = torch.arange(length - 1, device=device)
    visible[index, index + 1] = False
    return ~visible


def build_model(checkpoint: dict) -> PlateOCR2D:
    """Rebuild the model class recorded in a training checkpoint."""
    model_class = PlateOCRParseq if checkpoint.get("decoder") == "parseq" else PlateOCR2D
    return model_class(**checkpoint["model_config"])


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


def sequence_loss(logits: Tensor, target_ids: Tensor, label_smoothing: float = 0.0) -> Tensor:
    """Teacher-forced cross-entropy; target IDs include BOS, text, EOS, and PAD."""
    expected = target_ids[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        expected.reshape(-1),
        ignore_index=PAD_INDEX,
        label_smoothing=label_smoothing,
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
