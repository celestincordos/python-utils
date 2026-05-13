import os

from google.oauth2 import service_account

GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
GOOGLE_CREDENTIALS = service_account.Credentials.from_service_account_file(  # type: ignore
    GOOGLE_APPLICATION_CREDENTIALS
)
