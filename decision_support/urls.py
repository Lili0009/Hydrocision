from django.urls import path
from . import views
from django.contrib import admin
from django.shortcuts import redirect
from django.http import HttpResponse

admin.site.site_header = 'Hydrocision Administration'
admin.site.site_title = 'Hydrocision'
admin.site.index_title = 'Hydrocision Site'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda request: redirect('admin/') if request.user.is_authenticated else redirect('admin/login/?next=/admin/')),
    path('dashboard/', views.Ab, name='dashboard'),
    path('forecast/', views.Ac, name='forecast'),
    path('business_zone/', views.Ad, name='business_zone'),
    path('img_map/', views.Ae, name='img_map'),
    path('get-current-datetime/', views.Af, name='get_current_datetime'),
    path('health/', lambda request: HttpResponse('OK', status=200), name='health_check'),  
]