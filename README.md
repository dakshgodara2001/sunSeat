# SunSeat

Recommends the shadiest seat in a car based on route, time of day, and sun position. Powered by Google Maps directions and solar geometry calculations.

## Setup

### Prerequisites

- Python 3.9+
- A Google Maps API key with the **Directions API** enabled

### Get API keys

**Google Maps**

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or select an existing one)
3. Enable the **Directions API** under APIs & Services
4. Create an API key under Credentials
5. (Optional) Restrict the key to the Directions API

**OpenWeatherMap** (optional — not yet integrated, but planned for cloud cover data)

1. Sign up at [openweathermap.org](https://openweathermap.org/api)
2. Copy your API key from the dashboard

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```
GOOGLE_MAPS_API_KEY=your_key_here
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open http://localhost:8000 for the web UI, or use the API directly.

## Run with Docker

```bash
docker compose up --build
```

The API and frontend will be available at http://localhost:8000.

## Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All tests use mocked API responses — no real API keys needed.

## API usage

### `POST /recommend`

Returns the best (shadiest) and worst (sunniest) seat for a trip.

**Request:**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "New York, NY",
    "destination": "Newark, NJ",
    "departure_time": "2024-06-21T08:00:00Z",
    "drive_side": "LHD"
  }'
```

**Response:**

```json
{
  "best_seat": "FL",
  "worst_seat": "RR",
  "scores": {
    "FL": 0.0,
    "FR": 3.1245,
    "RL": 0.0,
    "RR": 3.1245
  },
  "confidence": "high",
  "summary": "Front Left recommended. Rear Right gets ~12 min of direct sun."
}
```

**Fields:**

| Field | Description |
|-------|-------------|
| `origin` | Start address or `"lat,lng"` |
| `destination` | End address or `"lat,lng"` |
| `departure_time` | ISO 8601 datetime (naive = UTC) |
| `drive_side` | `"LHD"` (default) or `"RHD"` |
| `vehicle_type` | Informational, e.g. `"sedan"` |

**Seats:** `FL` = Front Left, `FR` = Front Right, `RL` = Rear Left, `RR` = Rear Right.

### `GET /health`

```bash
curl http://localhost:8000/health
```

Returns `{"status": "ok"}`.
