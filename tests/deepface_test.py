import pytest
import numpy as np
from deepface import DeepFace

def test_embedding_size():
    """Проверяем, что размерность эмбеддинга соответствует ожидаемой."""
    fake_image = np.zeros((160, 160, 3), dtype=np.uint8)
    
    embedding_objs = DeepFace.represent(
        img_path=fake_image, 
        model_name="Facenet", 
        enforce_detection=False
    )
    
    embedding = embedding_objs[0]["embedding"]
    
    assert isinstance(embedding, list)
    assert len(embedding) == 512
