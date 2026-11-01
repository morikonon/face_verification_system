import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

client = QdrantClient(path="./test_local_db")

def create_db(collection_name: str = "test_local_db", vector_size: int = 512):

	if not client.collection_exists(collection_name=collection_name):
		client.create_collection(
			collection_name=collection_name,
			vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
		)

	print(f"Collection {collection_name} is created...")

def add_vector(points, collection_name: str = "test_local_db"):
	operation_info = client.upsert(
		collection_name, wait=True, points=points
	)

	print(f"Vector data added to qdrant")

def search_vector(
	collection_name: str = "test_local_db",
	query_vector: list[float] = None,
	limit: int = 25,
	with_vectors: bool = True
):
	search_results = client.query_points(
		collection_name=collection_name,
		query=query_vector,
		limit=limit,
		with_vectors=with_vectors
	)

	return [{"id": data.id, "vector": data.vector, "score": data.score, "payload": data.payload} for data in search_results.points]