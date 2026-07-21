from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("tasks", "0002_task_workflow"),
    ]

    operations = [
        migrations.RenameField(
            model_name="task",
            old_name="docker_info",
            new_name="runtime_info",
        ),
    ]
