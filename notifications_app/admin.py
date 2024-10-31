from django.contrib import admin
from django_celery_beat.models import SolarSchedule, ClockedSchedule, IntervalSchedule, CrontabSchedule, PeriodicTask


# Register your models here.
from .models import BroadcastNotification

admin.site.unregister(SolarSchedule)
admin.site.unregister(ClockedSchedule)
admin.site.unregister(IntervalSchedule)
admin.site.unregister(CrontabSchedule)
admin.site.unregister(PeriodicTask)

class BroadcastNotificationAdmin(admin.ModelAdmin):
    list_display = ('message', 'broadcast_on')

admin.site.register(BroadcastNotification, BroadcastNotificationAdmin)
