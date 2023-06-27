from towhee import api_service

service = api_service.build_service(
    [
        (lambda x: x, '/echo'),
    ]
)
