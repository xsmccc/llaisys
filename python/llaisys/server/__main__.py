"""Launch the LLAISYS Chat Server.

Usage:
    python -m llaisys.server --model ./models/DeepSeek-R1-Distill-Qwen-1.5B
    python -m llaisys.server --model ./models/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --port 8080
    python -m llaisys.server --model ./models/DeepSeek-R1-Distill-Qwen-1.5B-INT8 --quantized
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="LLAISYS Chat Server (OpenAI-compatible API)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the model directory (e.g. models/DeepSeek-R1-Distill-Qwen-1.5B)",
    )
    parser.add_argument(
        "--device", default="nvidia",
        choices=["cpu", "nvidia", "metax", "tianshu"],
        help="Compute device (default: nvidia)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512,
        help="KV cache max sequence length (default: 512)",
    )
    parser.add_argument(
        "--quantized", action="store_true",
        help="Load INT8 quantized weights (quantized_weights.npz)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    # Import here so --help is fast
    import uvicorn
    from .app import app, load_model

    load_model(args.model, args.device, args.max_seq_len, args.quantized)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
