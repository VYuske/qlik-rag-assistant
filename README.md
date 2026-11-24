# Qlik RAG Assistant

Assistente de RAG (Retrieval-Augmented Generation) **local** sobre a documentação em PDF do Qlik.

O pipeline transforma PDFs em markdown, divide em chunks, gera embeddings com Ollama e persiste em ChromaDB. A recuperação por similaridade já funciona; a camada de geração (LLM/chat) é a próxima fase.

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

```bash
qlik-rag stats
qlik-rag ingest
qlik-rag ingest --force
qlik-rag query "create bookmark" -k 5
```

Ou, sem instalar o entrypoint:

```bash
python -m qlik_rag.cli.main query "create bookmark"
```

Coloque os PDFs de origem em `data/raw/qlik/`. Todos os `.pdf` do diretório são indexados.

## Docker

O container roda o app e precisa alcançar o Ollama que roda no host.

**macOS / Windows** — o Ollama precisa aceitar conexões externas:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
cp .env.example .env
docker compose build
docker compose run --rm app query "create bookmark"
```

**Linux** — o Ollama costuma escutar apenas em `127.0.0.1`; use o override que roda o container na mesma rede do host:

```bash
docker compose -f docker-compose.yml -f docker-compose.linux.yml build
docker compose -f docker-compose.yml -f docker-compose.linux.yml run --rm app stats
```

Comandos disponíveis dentro do container:

```bash
docker compose run --rm app stats
docker compose run --rm app ingest --force
docker compose run --rm app query "create bookmark"
```

O índice fica em `./data/vectorstore/chroma`, montado como volume.

> Se `docker compose build` falhar em `apt-get update` por timeout de rede, use `docker compose build --network=host`. É uma limitação de roteamento IPv6 do Docker, não do projeto.

## Estrutura

```
src/qlik_rag/
├── config/        settings e prompts
├── ingestion/     loader de PDF, splitters e ingestor
├── retrieval/     acesso ao vector store
├── services/      orquestração de ingestão e busca
├── cli/           entrypoint argparse
└── logging_setup.py
```

Regras de arquitetura, estilo e de trabalho (TDD, commits, delegação) estão em [AGENTS.md](AGENTS.md).

## Testes e lint

```bash
pytest -q
ruff check src tests
```

Os testes usam `FakeEmbeddings` e `tmp_path`, portanto não exigem rede nem Ollama.
