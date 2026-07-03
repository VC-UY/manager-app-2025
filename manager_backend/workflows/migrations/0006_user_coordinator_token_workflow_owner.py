# Generated manually for multi-manager isolation

from django.db import migrations, models
import django.db.models.deletion
import workflows.models


class Migration(migrations.Migration):

    dependencies = [
        ('workflows', '0005_alter_workflow_workflow_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='coordinator_token',
            field=models.TextField(blank=True, help_text='Token JWT coordinateur pour ce manager', null=True),
        ),
        migrations.AlterField(
            model_name='user',
            name='is_staff',
            field=models.BooleanField(default=False, help_text='Designates whether the user can log into this admin site.', verbose_name='staff status'),
        ),
        migrations.AlterField(
            model_name='user',
            name='is_superuser',
            field=models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status'),
        ),
        migrations.AlterField(
            model_name='workflow',
            name='owner',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='workflows.user'),
        ),
    ]
