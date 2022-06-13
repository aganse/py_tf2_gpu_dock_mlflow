import mlflow

mlflow.projects.run(
    'https://github.com/aganse/py_tf2_gpu_dock_mlflow',
    backend='local',
    synchronous=False,
    experiment_name='Test/Debug',
    parameters={
        'run_name': 'patch_camelyon',
        'batch_size': 128,
        'epochs': 15,
        'convolutions': 0,
        'training_samples': 260000,
        'validation_samples': 30000,
        'randomize_images': True
    })

# mlflow.projects.run(
#     'https://github.com/aganse/py_tf2_gpu_dock_mlflow',
#     backend='local',
#     synchronous=False,
#     parameters={
#         'batch_size': 32,
#         'epochs': 10,
#         'convolutions': 2,
#         'training_samples': 15000,
#         'validation_samples': 2000,
#         'randomize_images': False
#     })

# mlflow.projects.run(
#     'https://github.com/aganse/py_tf2_gpu_dock_mlflow',
#     backend='local',
#     synchronous=False,
#     parameters={
#         'batch_size': 32,
#         'epochs': 10,
#         'convolutions': 0,
#         'training_samples': 15000,
#         'validation_samples': 2000,
#         'randomize_images': False
#     })
