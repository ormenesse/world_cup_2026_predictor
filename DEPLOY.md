# Deploy do app Streamlit (`app/streamlit_app.py`)

## ⚠️ Vercel não roda Streamlit
Vercel é **serverless/edge** (funções stateless + estáticos). Streamlit é um
**servidor persistente com WebSocket por usuário** (é assim que a sessão, os
reruns e os "jogadores criados nesta sessão" funcionam). Não há runtime suportado
para isso na Vercel; os "hacks" (embrulhar `streamlit run` numa função) quebram no
WebSocket e nos limites de cold-start/timeout. **Não use Vercel para o app em si.**

Se você QUER um domínio na Vercel: hospede o Streamlit num dos hosts abaixo e
coloque na Vercel uma página estática com um `<iframe src="https://SEU-APP">`.
(Ver o fim deste arquivo.)

Em runtime o app só precisa de ~22 MB de dados (NÃO precisa do `fifa_aggregated.csv`
de 7 GB): `app/app_data/*.parquet`, `data/gold_fifa_partidas/part-0.parquet`, os 3
`*.lgb` + `*_meta.json` e os `world_cup_2026_*.csv`. Garanta que esses arquivos
estejam commitados no repositório.

---

## Opção 1 (recomendada): Streamlit Community Cloud — grátis
1. Suba o repositório no GitHub (com os arquivos de dados/modelo acima).
2. https://share.streamlit.io → **New app** → escolha o repo/branch.
3. **Main file path**: `football_analysis/app/streamlit_app.py`
4. **Python version**: 3.12. **Requirements**: ele acha `football_analysis/requirements.txt`
   (se o seu repo tiver outra raiz, deixe um `requirements.txt` na raiz do repo também).
5. Deploy. Pronto — URL pública, sem servidor p/ gerenciar.

> Os "jogadores criados" já ficam só no navegador de cada visitante (session_state),
> então é seguro para um app público compartilhado.

## Opção 2: container (Render / Railway / Fly.io / Hugging Face Spaces)
Já existe `Dockerfile` + `.dockerignore` (imagem enxuta, exclui os dados pesados).

- **Render**: New → Web Service → repo → Environment **Docker** → root dir
  `football_analysis` → cria sozinho. (Render injeta `$PORT`, já tratado no CMD.)
- **Railway**: New Project → Deploy from repo → detecta o Dockerfile.
- **Fly.io**: `cd football_analysis && fly launch` (usa o Dockerfile) → `fly deploy`.
- **Hugging Face Spaces**: crie um Space tipo **Docker**, suba o conteúdo de
  `football_analysis/`.

Rodar a imagem localmente p/ testar:
```bash
cd football_analysis
docker build -t wc-sim .
docker run -p 8501:8501 wc-sim     # abre http://localhost:8501
```

## Opção 3: só quero a URL na Vercel
Hospede pela Opção 1/2 e publique na Vercel um `index.html` estático:
```html
<!doctype html><meta charset="utf-8"><title>WC Simulator</title>
<style>html,body,iframe{margin:0;height:100%;width:100%;border:0}</style>
<iframe src="https://SEU-APP.streamlit.app" allow="fullscreen"></iframe>
```
(Streamlit Community Cloud permite embed; em outros hosts garanta que não bloqueiam
iframe via `X-Frame-Options`.)
