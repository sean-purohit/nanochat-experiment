# API Server Guide

Complete guide for running the Inference API server and making requests.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/sot/Documents/26TBOT/nanochat
pip3 install fastapi uvicorn pydantic requests torch
```

### 2. Start Server

```bash
python3 -m experiment.server --checkpoint step_002920 --port 8000
```

Server will start at: **http://127.0.0.1:8000**

### 3. Test API

Open in browser: **http://127.0.0.1:8000/docs** for interactive API docs

Or use curl:
```bash
curl http://127.0.0.1:8000/health
```

## 📡 API Endpoints

### **POST /generate** - Generate Text

Generate text from a prompt with full control over sampling parameters.

**Request Body:**
```json
{
  "prompt": "123+456=",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": null,
  "stop_on_newline": false
}
```

**Parameters:**
- `prompt` (string, required): Input text to continue from
- `max_tokens` (integer, optional): Maximum tokens to generate (1-2048, default: 100)
- `temperature` (float, optional): Sampling temperature (0.0-2.0, default: 0.8)
  - 0.0 = Greedy (deterministic)
  - 0.8 = Balanced
  - 1.5+ = Creative/random
- `top_k` (integer, optional): Top-k sampling (1-32, default: null)
  - null = Use all tokens
  - 10 = Only sample from top 10 tokens
- `stop_on_newline` (boolean, optional): Stop generation at newline (default: false)

**Response:**
```json
{
  "prompt": "123+456=",
  "generated_text": "123+456=579",
  "tokens_generated": 3,
  "model": "step_002920"
}
```

**Example with curl:**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "123+456=",
    "max_tokens": 20,
    "temperature": 0.0
  }'
```

---

### **GET /health** - Health Check

Check if server and model are running.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "checkpoint": "step_002920"
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/health
```

---

### **GET /info** - Model Information

Get detailed information about the loaded model.

**Response:**
```json
{
  "checkpoint": "step_002920",
  "step": 2920,
  "val_loss": 0.2627,
  "model_layers": 24,
  "model_dim": 1536,
  "vocab_size": 32
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/info
```

---

### **GET /** - Root

Get API overview and available endpoints.

**Response:**
```json
{
  "service": "Inference API",
  "version": "1.0.0",
  "checkpoint": "step_002920",
  "status": "running",
  "endpoints": {
    "POST /generate": "Generate text from prompt",
    "GET /health": "Health check",
    "GET /info": "Model information",
    "GET /docs": "Interactive API documentation"
  }
}
```

---

### **GET /docs** - Interactive Documentation

Automatic interactive API documentation (Swagger UI).

Open in browser: **http://127.0.0.1:8000/docs**

Features:
- Try out all endpoints
- See request/response schemas
- Download OpenAPI spec

---

## 💻 Using the API

### Python Client

Use the provided client:

```python
from experiment.client_example import InferenceClient

# Create client
client = InferenceClient("http://127.0.0.1:8000")

# Generate text
result = client.generate(
    prompt="123+456=",
    max_tokens=20,
    temperature=0.0
)

print(result['generated_text'])
```

Run example:
```bash
python3 experiment/client_example.py
```

### curl Examples

**Greedy decoding (deterministic):**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "123+",
    "max_tokens": 30,
    "temperature": 0.0
  }'
```

**Sampling with temperature:**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "H+B=",
    "max_tokens": 50,
    "temperature": 0.8,
    "top_k": 10
  }'
```

**Stop on newline:**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "123\n456\n",
    "max_tokens": 100,
    "temperature": 0.5,
    "stop_on_newline": true
  }'
```

### JavaScript/TypeScript

```javascript
const response = await fetch('http://127.0.0.1:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: '123+456=',
    max_tokens: 20,
    temperature: 0.0
  })
});

const result = await response.json();
console.log(result.generated_text);
```

---

## 🔧 Server Configuration

### Command Line Options

```bash
python3 -m experiment.server [OPTIONS]

Required:
  --checkpoint FOLDER       Checkpoint folder name (e.g., step_002920)

Optional:
  --host HOST               Host to bind to (default: 127.0.0.1)
  --port PORT               Port to bind to (default: 8000)
  --device DEVICE           Device: cpu/cuda/mps (default: auto-detect)
  --reload                  Enable auto-reload (dev mode)
```

### Examples

**Start on default port:**
```bash
python3 -m experiment.server --checkpoint step_002920
```

**Start on custom port:**
```bash
python3 -m experiment.server --checkpoint step_002920 --port 8080
```

**Allow external connections:**
```bash
python3 -m experiment.server --checkpoint step_002920 --host 0.0.0.0 --port 8000
```

**Force CPU (slower but less memory):**
```bash
python3 -m experiment.server --checkpoint step_002920 --device cpu
```

**Development mode (auto-reload):**
```bash
python3 -m experiment.server --checkpoint step_002920 --reload
```

---

## 📊 Testing Different Checkpoints

Compare outputs from different checkpoints:

**Terminal 1 - Best checkpoint:**
```bash
python3 -m experiment.server --checkpoint step_002920 --port 8000
```

**Terminal 2 - Later checkpoint:**
```bash
python3 -m experiment.server --checkpoint step_003650 --port 8001
```

**Terminal 3 - Test both:**
```bash
# Test checkpoint 2920
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"prompt":"123+","max_tokens":20,"temperature":0.0}'

# Test checkpoint 3650
curl -X POST http://127.0.0.1:8001/generate -H "Content-Type: application/json" -d '{"prompt":"123+","max_tokens":20,"temperature":0.0}'
```

---

## 🎯 Best Practices

### Temperature Guidelines

| Task | Temperature | Why |
|------|-------------|-----|
| Math problems | 0.0 - 0.2 | Need exact answers |
| Structured output | 0.3 - 0.5 | Some creativity, mostly consistent |
| General text | 0.6 - 0.8 | Balanced variety |
| Creative/exploration | 0.9 - 1.5 | Maximum diversity |

### Top-K Guidelines

| Setting | Effect |
|---------|--------|
| `null` | Use full distribution (default) |
| 5-10 | Very focused, consistent |
| 20-40 | Balanced variety |
| 50+ | Similar to no top-k |

### Performance Tips

1. **Use temperature 0.0** for fastest generation (no sampling overhead)
2. **Reduce max_tokens** if you know expected length
3. **Use GPU** if available (`--device cuda` or `--device mps`)
4. **Keep server running** instead of restarting (model stays loaded)

---

## 🐛 Troubleshooting

### "Connection refused"
- Server not running - start with `python3 -m experiment.server --checkpoint step_002920`
- Wrong port - check port number in URL

### "Model not loaded" (503 error)
- Server still starting - wait a few seconds
- Check server logs for errors

### Slow generation
- Using CPU instead of GPU - check server logs
- Large max_tokens - reduce if possible
- Use `--device cuda` or `--device mps` when starting server

### Out of memory
- Model too large for GPU - use `--device cpu`
- Try smaller checkpoint (step_002920 = 2.5GB)
- Reduce max_tokens in request

### Wrong/unexpected outputs
- Check temperature (0.0 for deterministic)
- Check which checkpoint is loaded (`/info` endpoint)
- Verify prompt matches training data format

---

## 📈 Monitoring

### Server Logs

Server prints useful information:
```
INFO:     Started server process
INFO:     Waiting for application startup.
Loading model from checkpoint: step_002920
Model loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Request Logs

Each request is logged:
```
INFO:     127.0.0.1:52345 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:52346 - "GET /health HTTP/1.1" 200 OK
```

---

## 🔐 Production Deployment

For production use:

1. **Use proper ASGI server:**
   ```bash
   uvicorn experiment.server:app --host 0.0.0.0 --port 8000 --workers 1
   ```

2. **Add authentication** (optional):
   - Implement API key middleware
   - Use OAuth2/JWT tokens

3. **Add rate limiting:**
   - Use slowapi or similar
   - Limit requests per user/IP

4. **Monitor performance:**
   - Log response times
   - Track error rates
   - Monitor GPU/CPU usage

5. **Use reverse proxy:**
   - nginx or Caddy
   - SSL/TLS termination
   - Load balancing (if multiple servers)

---

## 📚 Additional Resources

- **Interactive Docs**: http://127.0.0.1:8000/docs
- **OpenAPI Spec**: http://127.0.0.1:8000/openapi.json
- **Client Example**: `experiment/client_example.py`
- **Inference Guide**: `experiment/INFERENCE_GUIDE.md`

---

## 🎉 Quick Test Script

Save as `test_api.sh`:

```bash
#!/bin/bash

# Test API
echo "Testing Inference API..."
echo ""

# Health check
echo "1. Health check:"
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
echo ""

# Model info
echo "2. Model info:"
curl -s http://127.0.0.1:8000/info | python3 -m json.tool
echo ""

# Generate text
echo "3. Generate text:"
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"123+456=","max_tokens":20,"temperature":0.0}' | python3 -m json.tool
echo ""

echo "Done!"
```

Run with: `chmod +x test_api.sh && ./test_api.sh`

---

Happy serving! 🚀
