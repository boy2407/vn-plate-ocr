# Layout-free plate OCR — 2D Transformer encoder-decoder

This model reads the complete plate crop and full transcription. It does not require a separate `single`/`double` layout label, line crops, or character boxes.

```mermaid
flowchart TD
    IMG["Full plate crop<br/>RGB · B × 3 × 64 × 128"]
    CNN["Residual CNN + BlurPool<br/>B × 256 × 4 × 32"]
    PROJ["Project each spatial cell<br/>256 → 192"]
    POS["Add row and column positional embeddings<br/>4 × 32 = 128 visual tokens"]
    ENC["Transformer visual encoder ×3<br/>4 heads · 192 dimensions<br/>B × 128 × 192"]
    DECIN["Teacher-forced text input<br/>BOS + prior characters"]
    DEC["Autoregressive Transformer decoder ×2<br/>causal self-attention + cross-attention<br/>to all 128 visual tokens"]
    HEAD["Dense vocabulary head<br/>PAD, BOS, EOS, digits, A–Z, Đ"]
    OUT["Greedy generation<br/>stop at EOS"]
    TEXT["Full plate transcription"]
    IMG --> CNN --> PROJ --> POS --> ENC
    DECIN --> DEC
    ENC --> DEC --> HEAD --> OUT --> TEXT
```

Training labels are full plate strings. Encode each as `BOS + characters + EOS`, pad each batch, and apply teacher-forced cross-entropy while ignoring PAD. The decoder can attend to all image rows at every character step, so it can learn top-row-then-bottom-row output without an explicit layout target. This is an architectural hypothesis; it must be trained and evaluated on both layouts before claiming it works.

Current interface in `custom_alpr/custom_ocr.py`: `PlateOCR2D(images, decoder_input_ids)` returns `[batch, steps, vocabulary]`; `generate(images)` emits strings. A dataset loader, training loop, checkpoint management, and pipeline adapter are not implemented yet.

See [decoder flow](decoder-flow.md) for the query/key/value shapes, four attention heads, Dense classifier, and training behavior.
