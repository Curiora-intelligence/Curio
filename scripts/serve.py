"""Offline local launcher. Run Alembic before starting; one worker on this Mac."""
import argparse
import os
from dotenv import load_dotenv
from app.core.models import cached_model


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    for mode, repo in [('TEXT', 'mlx-community/gpt-oss-20b-MXFP4-Q8'), ('VISION', 'mlx-community/Qwen3-VL-8B-Instruct-8bit')]:
        if not os.getenv('CURIO_MLX_' + mode + '_MODEL'):
            os.environ['CURIO_MLX_' + mode + '_MODEL'] = cached_model(repo)
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=args.port, workers=1, ws_max_size=740000)


if __name__ == '__main__':
    main()
