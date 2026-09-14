import argparse
import logging

from webcam.config import ModelPaths


def main():
    parser = argparse.ArgumentParser(description="EditaLive webcam")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--second-device", type=int,
                        help="Card for pose extraction and the VAE when dual GPU is selected "
                             "(default: --device + 1).")
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    parser.add_argument("--ckpt-dir", default=ModelPaths().checkpoint)
    parser.add_argument("--lora-paths", nargs="+", default=ModelPaths().loras)
    parser.add_argument("--fast-decoder-pth", default=ModelPaths().fast_decoder)
    parser.add_argument("--compile-cache-dir", default=ModelPaths().compile_cache)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import uvicorn
    from webcam.server import create_app
    paths = ModelPaths(checkpoint=args.ckpt_dir, loras=args.lora_paths,
                       fast_decoder=args.fast_decoder_pth, compile_cache=args.compile_cache_dir,
                       device=args.device, second_device=args.second_device)
    uvicorn.run(create_app(paths), host=args.host, port=args.port,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile,
                ws_max_size=17 * 1024 * 1024, ws_max_queue=2, ws_per_message_deflate=False)


if __name__ == "__main__":
    main()
