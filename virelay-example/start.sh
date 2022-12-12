python3 make_test_data_custom.py

python3 meta_analysis_custom.py test-project/attribution.h5 test-project/analysis.h5 --label-map test-project/label-map.json

python3 make_project.py \
    test-project/input.h5 \
    test-project/attribution.h5 \
    test-project/analysis.h5 \
    test-project/label-map.json \
    --project-name 'Test Project' \
    --dataset-name 'Random Data' \
    --model-name 'No Model' \
    --attribution-name 'Random Attribution' \
    --analysis-name 'Spectral Analysis' \
    --output test-project/project.yaml

gunicorn -w 4 -b 127.0.0.1:8080 \
    "virelay.application:create_app(projects=['test-project/project.yaml'])"
