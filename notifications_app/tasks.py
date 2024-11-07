from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import json
from celery.exceptions import Ignore
from .models import BroadcastNotification

@shared_task(bind=True)
def broadcast_notification(self, data):
    print(data)
    try:
        notification = BroadcastNotification.objects.filter(id=int(data)).first()

        if notification:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "notification_broadcast",  
                {
                    'type': 'send_notification',  
                    'message': json.dumps(notification.message), 
                }
            )
            notification.sent = True
            notification.save()
            print(f"Sending notification to group: notification_broadcast with message: {notification.message}")
            return 'Notification sent successfully'

        else:
            self.update_state(
                state='FAILURE',
                meta={'error': 'Notification not found'}
            )
            raise Ignore()

    except Exception as ex:
        self.update_state(
            state='FAILURE',
            meta={'error': str(ex)}
        )
        raise Ignore()
