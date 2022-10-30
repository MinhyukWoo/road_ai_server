import io
from django.shortcuts import render
from django import http

# Create your views here.
from rest_framework import generics
from ai_rest.serializers import PrecitSerializer
import tensorflow as tf
from rest_framework.response import Response

from PIL import Image
import numpy as np

model = tf.keras.models.load_model("final.h5")
from rest_framework import status
from rest_framework.parsers import MultiPartParser
import os


class PredictView(generics.CreateAPIView):
    serializer_class = PrecitSerializer
    name = "predict"

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        image_path = os.path.join("media", serializer.data["image"].split("/")[-1])
        image = Image.open(image_path).resize((150, 150)).convert("RGB")
        image_arr = np.array(image)[np.newaxis, ...]
        predictions = model.predict(image_arr)
        prediction_num = np.argmax(predictions)

        return Response(
            prediction_num, status=status.HTTP_201_CREATED, headers=headers
        )
