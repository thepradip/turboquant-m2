# Run TurboQuant Experiment — Step by Step

## Step 1: Create clean environment

```bash
python -m venv /tmp/tq_real_test
source /tmp/tq_real_test/bin/activate
```

## Step 2: Install

```bash
pip install /Users/pradip/Desktop/Learning/Claude/LLM_in_Prod/turboquant mlx mlx-lm psutil
```

## Step 3: Verify

```bash
python -c "import turboquant; print(turboquant.__version__)"
python -c "import mlx_lm; print('OK')"
```

## Step 4: Run what you need

```bash
cd /Users/pradip/Desktop/Learning/Claude/LLM_in_Prod/turboquant
```

### Chatbot (multi-turn, context grows each turn)

```bash
python examples/15_real_app.py --mode chatbot
```

### Long Document QA (paste document, ask questions)

```bash
python examples/15_real_app.py --mode qa
```

### RAG (retrieved chunks stuffed into context)

```bash
python examples/15_real_app.py --mode rag
```

### Interactive (type your own prompts)

```bash
python examples/15_real_app.py --mode interactive
```

### All modes at once

```bash
python examples/15_real_app.py --mode all
```

### Benchmark report (speed/memory at 1K-16K context)

```bash
python examples/14_mlx_full_report.py --contexts "1024,2048,4096,8192,16384"
```

## Step 5: Change model

```bash
python examples/15_real_app.py --model Qwen/Qwen2.5-1.5B-Instruct --mode qa
python examples/15_real_app.py --model mlx-community/Qwen3.5-2B-4bit --mode chatbot
```

## Step 6: Done

```bash
deactivate
```

## One command to run everything

```bash
python -m venv /tmp/tq_real_test && source /tmp/tq_real_test/bin/activate && pip install /Users/pradip/Desktop/Learning/Claude/LLM_in_Prod/turboquant mlx mlx-lm psutil && cd /Users/pradip/Desktop/Learning/Claude/LLM_in_Prod/turboquant && python examples/15_real_app.py --mode all
```
