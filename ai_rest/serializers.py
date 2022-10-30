from rest_framework import serializers

from ai_rest.models import PredictImage


class PrecitSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only=True)

    class Meta:
        model = PredictImage
        fields = (
            "id",
            "image",
        )
