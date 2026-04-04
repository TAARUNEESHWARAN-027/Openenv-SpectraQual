import os
import traceback


def main() -> None:
    try:
        from src.app import demo

        port = int(os.getenv("PORT", "7860"))
        print(f"[startup] launching Gradio on 0.0.0.0:{port}", flush=True)
        demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)
    except Exception:
        print("[startup] fatal error while starting app", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
