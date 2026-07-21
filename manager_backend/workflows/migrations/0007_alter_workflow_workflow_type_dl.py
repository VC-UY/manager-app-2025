from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('workflows', '0006_user_coordinator_token_workflow_owner'),
    ]

    operations = [
        migrations.AlterField(
            model_name='workflow',
            name='workflow_type',
            field=models.CharField(
                choices=[
                    ('MATRIX_ADDITION', 'Addition de matrices de grande taille'),
                    ('MATRIX_MULTIPLICATION', 'Multiplication de matrices de grande taille'),
                    ('ML_TRAINING', 'Entraînement de modèle machine learning'),
                    ('ML_INFERENCE', 'Inférence de modèle machine learning'),
                    ('OPEN_MALARIA', 'Simulation de la propagation du paludisme'),
                    ('DISTRIBUTED_LEARNING', 'Apprentissage distribué gossip (AD-PSGD)'),
                    ('CUSTOM', 'Workflow personnalisé'),
                ],
                default='MATRIX_ADDITION',
                max_length=30,
            ),
        ),
    ]
