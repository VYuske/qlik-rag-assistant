# AGENTS.md

Regras de trabalho para agentes (humanos ou IA) neste repositório.

---

## 1. Escopo do projeto

**qlik-rag-assistant** é um assistente de RAG (Retrieval-Augmented Generation) **local** sobre a documentação em PDF do Qlik.

Pipeline implementado hoje:

```
PDF (data/raw/qlik)
  -> OpenDataLoaderPDFLoader (PDF -> markdown)
  -> MarkdownHeaderTextSplitter (split por ###/####/#####)
  -> MarkdownTextSplitter (chunks de tamanho fixo)
  -> OllamaEmbeddings (modelo local)
  -> ChromaDB (persistência em data/vectorstore/chroma)
  -> retriever (busca por similaridade)
  -> grafo LangGraph (retrieve -> grade -> reformulate* -> generate)
  -> chat (Streamlit ou CLI)
```

Estado atual:

- **Existe**: configuração, ingestão, embeddings, vector store, recuperação, grafo de geração com loop de avaliação, CLI (`stats`, `ingest`, `query`, `ask`) e UI Streamlit com streaming e fontes.
- **Não existe ainda**: memória de conversa persistente entre sessões (hoje o histórico vive no `st.session_state`) e avaliação automatizada da qualidade das respostas.

Stack: Python 3.12, LangChain, LangGraph, Ollama (local), ChromaDB, Streamlit, `pydantic-settings`, `pytest`, `ruff`, Docker.

Tudo roda **offline**. Modelos são servidos pelo Ollama no host (`http://localhost:11434` fora do container, `http://host.docker.internal:11434` dentro).

---

## 2. Arquitetura e regra de dependência

```
src/qlik_rag/
├── config/        settings.py (Settings), prompts.py (SYSTEM_PROMPT e prompts do grafo)
├── ingestion/     loaders.py, splitters.py, ingestor.py
├── retrieval/     vector_store.py (VectorStoreService)
├── generation/    llm.py, state.py, graph.py (grafo RAG)
├── services/      ingestion_service.py, search_service.py, chat_service.py
├── ui/            app.py (Streamlit), formatting.py
├── cli/           main.py (entrypoint argparse)
└── logging_setup.py
```

Regra de dependência **unidirecional**:

```
ui -> services -> {generation, ingestion, retrieval} -> config
cli -> services -> ...
```

- `config` não importa nada do projeto (folha da árvore).
- `ingestion` e `retrieval` não conhecem `services` nem `ui`.
- `generation` recebe o retriever e o LLM por injeção; não instancia Chroma.
- `services` orquestra domínio; `cli` e `ui` apenas traduzem entrada para chamadas de serviço.
- `ui/` e `cli/` são as únicas camadas que escrevem em stdout/tela. Nada fora delas chama `sys.exit`, faz parsing de argumentos ou imprime.

Ao adicionar um módulo, escolha a camada pela dependência, não pela conveniência do import. Se a regra precisar mudar, atualize este arquivo no mesmo commit.

---

## 3. Regras de código

### 3.1 Comentários: somente no cabeçalho

- **Permitido**:
  - Docstring de módulo no topo do arquivo, descrevendo sua responsabilidade.
  - Docstring de classe ou função pública quando o contrato não for óbvio pela assinatura.
- **Proibido**:
  - Comentários `#` soltos dentro de funções, blocos `if`/`for` ou no meio do código.
  - Comentários de "seção" (`# --- CONFIGURAÇÕES ---`), código comentado e `# TODO` sem issue.
  - Cabeçalhos duplicados (`# scripts/ingest.py`) ou que não correspondem ao arquivo.

Se você sente necessidade de comentar um trecho, prefira: extrair uma função com nome descritivo, melhorar o nome da variável, ou mover a explicação para a docstring da função.

### 3.2 Estilo e tipagem

- Type hints obrigatórios em toda assinatura pública (parâmetros e retorno).
- `from __future__ import annotations` quando usar tipos modernos em runtime.
- Sem código morto: imports, variáveis, funções e arquivos não usados são removidos, não comentados.
- Sem `except Exception` que apenas imprime; capture o erro esperado e propague com contexto (`raise ... from exc`).
- `logging` (via `logging.getLogger(__name__)`) em vez de `print`. Diagnóstico de CLI pode usar stdout em `cli/`.
- Nomenclatura: módulos e funções `snake_case`, classes `PascalCase`, constantes `UPPER_SNAKE`, campos de `Settings` em `snake_case`.
- `ruff` deve passar sem erros; não use `# noqa` sem justificativa na mesma linha.
- `ruff format --check src tests` deve passar: o CI verifica formatação, então rode `ruff format src tests` antes de commitar.

### 3.3 Configuração

- Toda configuração entra por `Settings` (`pydantic-settings`), com default explícito e leitura de `.env`.
- Novas variáveis de ambiente devem ser adicionadas ao `.env.example` no mesmo commit.
- Não leia `os.getenv` fora de `config/settings.py`.

---

## 4. Fluxo de trabalho: teste antes de implementar

TDD é obrigatório para features e fixes:

1. **Vermelho**: escreva ou ajuste o teste que falha e representa o comportamento desejado.
2. **Verde**: implemente o mínimo para passar.
3. **Refatore**: limpe o código mantendo a suíte verde.
4. **Confirme** que a suíte completa passa antes de commitar.

Regras:

- Nenhuma feature ou fix sem teste correspondente em `tests/`.
- Testes não podem depender de rede, Ollama ou GPU. Use `langchain_core.embeddings.FakeEmbeddings`, `tmp_path` e doubles simples.
- Bug reportado = teste de regressão primeiro, que reproduz o defeito.
- Se um teste for inviável, registre o motivo no PR/commit e crie o equivalente mais próximo.

Comandos:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

---

## 5. Commits

- **Um commit por implementação concluída** (não acumule feature + fix + docs num commit só).
- Conventional Commits, mensagem em português:
  - `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `build:`, `perf:`
- Escopo opcional entre parênteses: `refactor(ingestion): isolar metadata por chunk`.
- Commits atômicos: refactor e mudança de comportamento em commits separados.
- Nunca commite `data/vectorstore/`, `__pycache__/`, `.env` ou artefatos de build.
- Só crie commits quando o trabalho estiver validado por testes.

---

## 6. Papéis e delegação

### 6.1 Agente principal (orquestrador)

Responsável por **conhecer e documentar o escopo** do projeto e por **delegar** o trabalho operacional.

Faz:

- Mantém este `AGENTS.md` e o `README.md` corretos e atualizados.
- Define arquitetura, contratos públicos e critérios de aceite (DoD).
- Decompõe o trabalho em tarefas pequenas, com arquivos-alvo e testes esperados.
- Decide o que é local (arquitetura, contratos, integração final) e o que é delegável.
- Revisa o resultado final e valida a suíte completa antes de fechar.

Não faz (delega):

- Implementação de módulos isolados e tarefas de um único arquivo.
- Refactors mecânicos e renomeações.
- Escrita inicial de testes e massa de dados de fixtures.
- Revisão de código detalhada linha a linha.

### 6.2 Agentes secundários

| Agente | Papel | Recebe do principal |
|---|---|---|
| `explore` | Mapear código, dependências e fluxos; responder "onde está X" | Pergunta delimitada e área do repositório |
| `plan` | Desenhar abordagem de implementação com trade-offs | Objetivo, restrições e achados do `explore` |
| `general` | Implementar tarefas pequenas/médias, escrever testes, executar refactors mecânicos | Escopo fechado, arquivos-alvo, DoD |
| revisor (via `general`) | Revisar diff antes do merge: estilo, testes, comentários soltos, código morto | Diff/commit e checklist deste arquivo |

### 6.3 Matriz de delegação

| Tarefa | Responsável |
|---|---|
| Definir/alterar arquitetura ou contrato público | Agente principal |
| Atualizar `AGENTS.md`, `README.md` e decisões de escopo | Agente principal |
| Explorar repositório e levantar contexto | Secundário (`explore`) |
| Desenhar plano de implementação | Secundário (`plan`), aprovação do principal |
| Implementar módulo novo ou feature isolada | Secundário (`general`) |
| Bug fix com teste de regressão | Secundário (`general`) |
| Refactor mecânico / renomeações | Secundário (`general`) |
| Revisão de diff e checklist de qualidade | Secundário revisor |
| Integração final, validação da suíte e commit | Agente principal |

### 6.4 Contrato de handoff

Ao **delegar**, informe sempre:

1. Objetivo em uma frase.
2. Arquivos-alvo (caminhos absolutos) e o que **não** pode ser alterado.
3. Restrições de estilo (seção 3) e comportamento esperado.
4. Testes que devem existir e como executá-los.
5. Formato de retorno esperado: resumo, lista de arquivos alterados, saída de `pytest`/`ruff`.

Ao **receber** um handoff:

1. Leia os arquivos-alvo antes de propor mudanças.
2. Escreva o teste que falha.
3. Implemente.
4. Rode `pytest` e `ruff` e reporte o resultado literal.
5. Não crie commits por conta própria se o principal não pediu; entregue o diff.

---

## 7. Definition of Done

Uma tarefa só está concluída quando:

- [ ] Existe teste cobrindo o comportamento novo/corrigido, e ele passa.
- [ ] `pytest -q`, `ruff check src tests` e `ruff format --check src tests` passam.
- [ ] O workflow de CI (`.github/workflows/ci.yml`) passaria com a mudança.
- [ ] Não há comentários soltos, código comentado, imports não usados ou arquivos mortos.
- [ ] Configuração nova está em `Settings` e no `.env.example`.
- [ ] `AGENTS.md`/`README.md` atualizados se o contrato ou o modo de uso mudou.
- [ ] Commit atômico criado com mensagem convencional.

---

## 8. Comandos

### Ambiente local

```bash
conda env create -f conda-enviroment.yml
conda activate lab-langgraph
pip install -e ".[dev]"
```

### Testes e lint

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

### CLI

```bash
qlik-rag stats
qlik-rag ingest
qlik-rag ingest --force
qlik-rag query "create bookmark" -k 5
qlik-rag ask "how do I create a bookmark?"
```

Equivalente sem instalação: `python -m qlik_rag.cli.main <comando>`.

### Chat

```bash
streamlit run src/qlik_rag/ui/app.py
```

### Docker

```bash
cp .env.example .env
docker compose build
docker compose run --rm app stats
docker compose run --rm app ingest --force
docker compose run --rm app query "create bookmark"
```

Em Linux, onde o Ollama escuta só em `127.0.0.1`, acrescente o override:

```bash
docker compose -f docker-compose.yml -f docker-compose.linux.yml run --rm app stats
```

### Pré-requisitos de modelos

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b
```

---

## 9. Higiene e segurança

- O índice vetorial (`data/vectorstore/`) é **artefato regenerável**: nunca versionar.
- PDFs de origem em `data/raw/` são versionados por serem a matéria-prima do RAG.
- `.env` nunca é commitado; use `.env.example` como referência.
- Processamento é 100% local. Não introduza APIs externas, telemetria ou envio de documentos para terceiros sem aprovação explícita.
- Não altere o modelo de embeddings sem planejar `ingest --force`, pois o índice existente fica incompatível.
