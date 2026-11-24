# Qlik RAG Assistant

Assistente de RAG (Retrieval-Augmented Generation) **local** sobre a documentação em PDF do Qlik.

O pipeline transforma PDFs em markdown, divide em chunks, gera embeddings com Ollama e persiste em ChromaDB. Sobre a recuperação há um grafo LangGraph com loop de avaliação e uma interface de chat em Streamlit.

## Requisitos

- Python 3.12
- [Conda](https://docs.conda.io/) (recomendado)
- [Ollama](https://ollama.com/) rodando localmente
- Java (JRE) — exigido pelo `OpenDataLoaderPDFLoader`

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b
```

## Instalação

```bash
conda env create -f conda-enviroment.yml
conda activate lab-langgraph
pip install -e ".[dev]"
cp .env.example .env
```

## Uso

### Chat no navegador

```bash
streamlit run src/qlik_rag/ui/app.py
```

Abre em `http://localhost:8501`. A resposta aparece em streaming, com as fontes em um expander, e a sidebar mostra a contagem de chunks e o botão de reindexação.

### Linha de comando

```bash
qlik-rag stats
qlik-rag ingest
qlik-rag ingest --force
qlik-rag query "create bookmark" -k 5
qlik-rag ask "how do I create a bookmark?"
```

Ou, sem instalar o entrypoint:

```bash
python -m qlik_rag.cli.main ask "how do I create a bookmark?"
```

Coloque os PDFs de origem em `data/raw/qlik/`. Todos os `.pdf` do diretório são indexados.

### Como a resposta é montada

O grafo (`src/qlik_rag/generation/graph.py`) executa:

```
retrieve -> grade -> (suficiente ou limite atingido?) -> generate
                  \-> (insuficiente) -> reformulate -> retrieve
```

- `MAX_RETRIEVAL_LOOPS` (default `1`) limita quantas vezes a pergunta pode ser reformulada.
- Se a avaliação falhar, o grafo segue direto para a resposta em vez de quebrar.
- Quando a documentação não cobre a pergunta, a resposta é a frase de recusa definida em `config/prompts.py`.

## Docker

O container roda o app e precisa alcançar o Ollama que roda no host.

**macOS / Windows** — o Ollama precisa aceitar conexões externas:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
cp .env.example .env
docker compose build
docker compose up ui
```

**Linux** — o Ollama costuma escutar apenas em `127.0.0.1`; use o override que roda o container na mesma rede do host:

```bash
docker compose -f docker-compose.yml -f docker-compose.linux.yml build
docker compose -f docker-compose.yml -f docker-compose.linux.yml up ui
```

A interface fica em `http://localhost:8501`. Para os comandos de CLI no container:

```bash
docker compose run --rm app stats
docker compose run --rm app ingest --force
docker compose run --rm app ask "how do I create a bookmark?"
```

O índice fica em `./data/vectorstore/chroma`, montado como volume.

> Se `docker compose build` falhar em `apt-get update` por timeout de rede, use `docker compose build --network=host`. É uma limitação de roteamento IPv6 do Docker, não do projeto.

## Estrutura

```
src/qlik_rag/
├── config/        settings, prompts
├── ingestion/     loader de PDF, splitters e ingestor
├── retrieval/     acesso ao vector store
├── generation/    modelo de chat, estado e grafo RAG
├── services/      ingestão, busca e chat
├── ui/            app Streamlit e formatação
├── cli/           entrypoint argparse
└── logging_setup.py
```

Regras de arquitetura, estilo e de trabalho (TDD, commits, delegação) estão em [AGENTS.md](AGENTS.md).

## Testes e lint

```bash
pytest -q
ruff check src tests
```

Os testes usam `FakeEmbeddings`, LLM falso e `tmp_path`, portanto não exigem rede nem Ollama.
