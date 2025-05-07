import requests

from django.conf import settings
from django.contrib.sites.shortcuts import get_current_site
from functools import wraps

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

DEFAULT_DISTRIBUTION = getattr(settings, "UESR_AGENT_DISTRIBUTION", "django-requests")


def user_agent(name):
    try:
        v = version(distribution_name=name)
    except PackageNotFoundError:
        v = "unknown"

    try:
        domain = get_current_site(None).domain
    except:
        domain = "unknown"
    ua = f"{name}/{v} (+{domain})"
    return ua


class DjangoSession(requests.Session):
    def __init__(self, distribution_name=DEFAULT_DISTRIBUTION):
        super().__init__()
        self.headers["user-agent"] = user_agent(distribution_name)


USER_AGENT = getattr(settings, "USER_AGENT", user_agent(DEFAULT_DISTRIBUTION))


@wraps(requests.request, assigned=["__doc__"])
def request(method, url, **kwargs):
    with DjangoSession() as session:
        return session.request(method=method, url=url, **kwargs)


@wraps(requests.get, assigned=["__doc__"])
def get(url, **kwargs) -> requests.Response:
    return request("get", url, **kwargs)


@wraps(requests.options, assigned=["__doc__"])
def options(url, **kwargs) -> requests.Response:
    return request("options", url, **kwargs)


@wraps(requests.post, assigned=["__doc__"])
def post(url, data=None, json=None, **kwargs) -> requests.Response:
    return request("post", url, data=data, json=json, **kwargs)


@wraps(requests.put, assigned=["__doc__"])
def put(url, data=None, **kwargs) -> requests.Response:
    return request("put", url, data=data, **kwargs)


@wraps(requests.patch, assigned=["__doc__"])
def patch(url, data=None, **kwargs) -> requests.Response:
    return request("patch", url, data=data, **kwargs)


@wraps(requests.delete, assigned=["__doc__"])
def delete(url, **kwargs) -> requests.Response:
    return request("delete", url, **kwargs)
