from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),          # Admin interface
    path("", include("app.urls")),            # Root URL maps to `app.urls`
]
