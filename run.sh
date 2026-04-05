#!/bin/bash
echo "Starting backend..."
cd drone-ui/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --port 8000 &
BACKEND_PID=$!

echo "Starting frontend..."
cd ../frontend
npm run dev -- --port 5173 &
FRONTEND_PID=$!

function cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
}

trap cleanup EXIT
wait
