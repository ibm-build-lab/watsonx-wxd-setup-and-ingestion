import asyncio
import json
import logging
import re
import os
import tempfile
import requests
from pathlib import Path
from typing import List, Dict, Optional

import aiohttp
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator  # type: ignore
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from tqdm import tqdm


logger = logging.getLogger(__name__)


class CloudObjectStorageReader(BaseReader):
    """
    A class used to interact with IBM Cloud Object Storage.

    This class inherits from the BasePydanticReader base class
    and overrides its methods to work with IBM Cloud Object Storage.

    Compatible with llama-index framework.

    Taken from wxd-setup-and-ingestion repository in skol-assets

    Attributes
    ----------
    bucket_name : str
        The name of the bucket in the cloud storage.
    credentials : dict
        The credentials required to authenticate with the cloud storage.
        It must contain 'apikey' and 'service_instance_id'.
    hostname : str, optional
    """

    def __init__(
        self,
        bucket_name: str,
        credentials: dict,
        hostname: str = "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        file_extractor: Optional[Dict[str, BaseReader]] = None,
        num_files_limit: Optional[int] = None,
    ):
        self.bucket_name = bucket_name
        self.credentials = credentials
        self.hostname = hostname
        self.num_files_limit = num_files_limit
        self.file_extractor = file_extractor
        self._base_url = f"{self.hostname}/{self.bucket_name}"
        self._authenticator = IAMAuthenticator(self.credentials["apikey"])
        if "apikey" in self.credentials and "service_instance_id" in self.credentials:
            self.credentials = credentials
        else:
            raise ValueError(
                "Missing 'apikey' or 'service_instance_id' in credentials."
            )

    @property
    def bearer_token(self):
        if not hasattr(self, "_bearer_token"):
            self._bearer_token = self._authenticator.token_manager.get_token()
        return self._bearer_token

    @property
    def session_headers(self):
        return {
            "ibm-service-instance-id": self.credentials["service_instance_id"],
            "Authorization": f"Bearer {self.bearer_token}",
        }

    @property
    def async_session(self):
        if not hasattr(self, "_async_session"):
            self._async_session = aiohttp.ClientSession(headers=self.session_headers)
        return self._async_session

    @property
    def session(self):
        if not hasattr(self, "_session"):
            self._session = requests.Session()
            self._session.headers.update(self.session_headers)
        return self._session

    def load_data(self, show_progress: bool = False) -> List[Document]:
        return asyncio.get_event_loop().run_until_complete(
            self.aload_data(show_progress)
        )

    async def aload_data(self, show_progress: bool = False) -> List[Document]:

        async def write_file_to_dir(file_name, dir, pbar: Optional[tqdm] = None):
            data = await self.aget_file_data(file_name)
            with open(os.path.join(dir, file_name), "wb") as f:
                f.write(data)
            if pbar is not None:
                pbar.update()
            return data

        file_names = self.list_files()
        with tempfile.TemporaryDirectory() as temp_dir:
            with tqdm(
                total=len(file_names),
                disable=not show_progress,
                desc="Downloading files to temp dir",
            ) as pbar:
                tasks = [
                    write_file_to_dir(file_name, temp_dir, pbar)
                    for file_name in file_names
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    if hasattr(self, "_async_session"):
                        await self._async_session.close()
                        del self._async_session
            reader = SimpleDirectoryReader(temp_dir, file_extractor=self.file_extractor)
            documents = reader.load_data(show_progress=show_progress)

        return documents

    def list_files(self) -> List[str]:
        """
        Retrieves a list of file names from the COS bucket.

        Returns:
            List[str]: A list of file names from the COS bucket.
        """
        response_text = self.session.get(self._base_url).text
        file_names = re.findall(r"<Key>(.*?)</Key>", response_text)
        return file_names

    async def aget_file_data(self, file_name: str) -> bytes:
        """
        Asynchronously retrieves the content of a file from the COS bucket.

        Args:
            file_name (str): The name of the file.

        Returns:
            bytes: The file data.
        """

        url = f"{self._base_url}/{file_name}"
        async with self.async_session.get(url) as response:
            data = await response.read()
        return data

    def get_file_data(self, file_name: str) -> bytes:
        """
        Retrieves the content of a file from the COS bucket.

        Args:
            file_name (str): The name of the file to retrieve.

        Returns:
            bytes: The content of the file as bytes.

        """
        url = f"{self._base_url}/{file_name}"
        response = self.session.get(url)
        return response.content

    @classmethod
    def from_service_credentials(
        cls,
        bucket: str,
        service_credentials_path: Path | str,
        hostname: str = "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        *,
        num_files_limit: Optional[int] = None,
        file_extractor: Optional[Dict[str, BaseReader]] = None,
    ) -> "CloudObjectStorageReader":
        with open(service_credentials_path, "r") as file:
            cos_auth_dict = json.load(file)
        credentials = {
            "apikey": cos_auth_dict["apikey"],
            "service_instance_id": cos_auth_dict["resource_instance_id"],
        }
        return cls(
            bucket_name=bucket,
            credentials=credentials,
            hostname=hostname,
            num_files_limit=num_files_limit,
            file_extractor=file_extractor,
        )
