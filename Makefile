.PHONY: proto back service test

proto:
	uv run python -m grpc_tools.protoc -I backend/proto \
	  --python_out=backend/generated \
	  --grpc_python_out=backend/generated \
	  backend/proto/chat.proto
	sed -i 's/^import chat_pb2/from backend.generated import chat_pb2/' \
	  backend/generated/chat_pb2_grpc.py

back:
	uv run uvicorn backend.orchestrator.main:app --reload

service:
	uv run python -m backend.service.main

test:
	uv run pytest