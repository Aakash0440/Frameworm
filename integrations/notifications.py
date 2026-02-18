"""
Notification integrations for training alerts.

Example:
    >>> from frameworm.integrations import SlackNotifier
    >>> 
    >>> slack = SlackNotifier(webhook_url='https://hooks.slack.com/...')
    >>> trainer.add_callback(slack)
    >>> # Get notified when training completes or fails
"""

from typing import Optional
from training.callbacks import Callback
import requests


class SlackNotifier(Callback):
    """
    Send Slack notifications on training events.
    
    Args:
        webhook_url: Slack webhook URL
        notify_on_start: Send notification on train start (default: False)
        notify_on_end: Send notification on train end (default: True)
        notify_on_epoch: Send updates every N epochs (default: None)
        
    Example:
        >>> slack = SlackNotifier(
        ...     webhook_url='https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        ...     notify_on_epoch=10
        ... )
        >>> trainer.add_callback(slack)
    """
    
    def __init__(
        self,
        webhook_url: str,
        notify_on_start: bool = False,
        notify_on_end: bool = True,
        notify_on_epoch: Optional[int] = None
    ):
        self.webhook_url = webhook_url
        self.notify_on_start = notify_on_start
        self.notify_on_end = notify_on_end
        self.notify_on_epoch = notify_on_epoch
    
    def _send_message(self, message: str):
        """Send message to Slack"""
        try:
            response = requests.post(
                self.webhook_url,
                json={'text': message}
            )
            response.raise_for_status()
        except Exception as e:
            print(f"⚠️  Failed to send Slack notification: {e}")
    
    def on_train_begin(self, trainer):
        if self.notify_on_start:
            self._send_message("🚀 Training started!")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        if self.notify_on_epoch and (epoch + 1) % self.notify_on_epoch == 0:
            metrics_str = ', '.join(f"{k}={v:.4f}" for k, v in metrics.items())
            self._send_message(f"📊 Epoch {epoch + 1}: {metrics_str}")
    
    def on_train_end(self, trainer):
        if self.notify_on_end:
            final_metrics = trainer.state.val_metrics
            if final_metrics:
                metrics_str = ', '.join(f"{k}={v[-1]:.4f}" for k, v in final_metrics.items() if v)
                self._send_message(f"✅ Training complete! Final: {metrics_str}")
            else:
                self._send_message("✅ Training complete!")


class EmailNotifier(Callback):
    """
    Send email notifications on training events.
    
    Args:
        smtp_server: SMTP server address
        smtp_port: SMTP port (default: 587)
        sender_email: Sender email address
        sender_password: Sender password
        recipient_email: Recipient email address
        
    Example:
        >>> email = EmailNotifier(
        ...     smtp_server='smtp.gmail.com',
        ...     sender_email='your@gmail.com',
        ...     sender_password='app_password',
        ...     recipient_email='notify@example.com'
        ... )
        >>> trainer.add_callback(email)
    """
    
    def __init__(
        self,
        smtp_server: str,
        sender_email: str,
        sender_password: str,
        recipient_email: str,
        smtp_port: int = 587
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
    
    def _send_email(self, subject: str, body: str):
        """Send email notification"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
        
        except Exception as e:
            print(f"⚠️  Failed to send email: {e}")
    
    def on_train_end(self, trainer):
        subject = "FRAMEWORM Training Complete"
        body = f"Your training job has completed.\n\nFinal metrics:\n"
        
        for key, values in trainer.state.val_metrics.items():
            if values:
                body += f"  {key}: {values[-1]:.4f}\n"
        
        self._send_email(subject, body)