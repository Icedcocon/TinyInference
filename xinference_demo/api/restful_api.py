import asyncio
import inspect
import json
import logging
import os
import pprint
import sys
import warnings
from typing import Optional, Any

import xoscar as xo
from aioprometheus import REGISTRY, MetricsMiddleware
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse as StarletteJSONResponse # starlette 是 fastAPI 的组件
from starlette.responses import RedirectResponse
from uvicorn import Config, Server
from xoscar.utils import get_next_port

from ..constants import XINFERENCE_DEFAULT_ENDPOINT_PORT
from ..core.supervisor import SupervisorActor
from ..core.utils import json_dumps


logger = logging.getLogger(__name__)

class JSONResponse(StarletteJSONResponse):  # type: ignore # noqa: F811
    '''
    对 starlette.responses.JSONResponse 封装
    添加 render 函数用于 dump 结构体
    '''
    def render(self, content: Any) -> bytes:
        return json_dumps(content)

class RESTfulAPI:
    '''
    创建并管理 FastAPI 和 APIRouter 对象
    添注册路由函数并定义对应 handler
    '''

    def __init__(
        self,
        supervisor_address: str,
        host: str,
        port: int,
        auth_config_file: Optional[str] = None,
    ):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._host = host
        self._port = port
        self._supervisor_ref = None
        # self._auth_config: AuthStartupConfig = self.init_auth_config(auth_config_file)
        self._auth_config = True
        self._router = APIRouter()
        self._app = FastAPI()

    def is_authenticated(self):
        return False if self._auth_config is None else True
    
    async def _get_supervisor_ref(self) -> xo.ActorRefType[SupervisorActor]:
        if self._supervisor_ref is None:
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.uid()
            )
        return self._supervisor_ref

    def serve(self, logging_conf: Optional[dict] = None):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # internal interface
        self._router.add_api_route("/status", self.get_status, methods=["GET"])
        # conflict with /v1/models/{model_uid} below, so register this first
        self._router.add_api_route(
            "/v1/models/prompts", self._get_builtin_prompts, methods=["GET"]
        )
        self._router.add_api_route(
            "/v1/models/families", self._get_builtin_families, methods=["GET"]
        )
        self._router.add_api_route(
            "/v1/cluster/devices", self._get_devices_count, methods=["GET"]
        )
        self._router.add_api_route("/v1/address", self.get_address, methods=["GET"])

        # Clear the global Registry for the MetricsMiddleware, or
        # the MetricsMiddleware will register duplicated metrics if the port
        # conflict (This serve method run more than once).
        REGISTRY.clear()
        self._app.add_middleware(MetricsMiddleware)
        self._app.include_router(self._router)

        # 检查路由返回的 Response 类型是否合法.
        # This is to avoid `jsonable_encoder` performance issue:
        # https://github.com/xorbitsai/inference/issues/647
        invalid_routes = []
        try:
            for router in self._router.routes:
                return_annotation = router.endpoint.__annotations__.get("return")
                if not inspect.isclass(return_annotation) or not issubclass(
                    return_annotation, Response
                ):
                    invalid_routes.append(
                        (router.path, router.endpoint, return_annotation)
                    )
        except Exception:
            pass  # In case that some Python version does not have __annotations__
        if invalid_routes:
            raise Exception(
                f"The return value type of the following routes is not Response:\n"
                f"{pprint.pformat(invalid_routes)}"
            )
        
        # class SPAStaticFiles(StaticFiles):
        #     '''
        #     对 fastapi.staticfiles.StaticFiles 封装，实现 SPA 效果
        #     '''
        #     async def get_response(self, path: str, scope):
        #         response = await super().get_response(path, scope)
        #         if response.status_code == 404:
        #             response = await super().get_response(".", scope)
        #         return response

        # try:
        #     package_file_path = __import__("xinference").__file__
        #     assert package_file_path is not None
        #     lib_location = os.path.abspath(os.path.dirname(package_file_path))
        #     ui_location = os.path.join(lib_location, "web/ui/build/")
        # except ImportError as e:
        #     raise ImportError(f"Xinference is imported incorrectly: {e}")

        # if os.path.exists(ui_location):

        #     @self._app.get("/")
        #     def read_main():
        #         response = RedirectResponse(url="/ui/")
        #         return response

        #     self._app.mount(
        #         "/ui/",
        #         SPAStaticFiles(directory=ui_location, html=True),
        #     )
        # else:
        #     warnings.warn(
        #         f"""
        #     Xinference ui is not built at expected directory: {ui_location}
        #     To resolve this warning, navigate to {os.path.join(lib_location, "web/ui/")}
        #     And build the Xinference ui by running "npm run build"
        #     """
        #     )

        config = Config(
            app=self._app, host=self._host, port=self._port, log_config=logging_conf
        )
        server = Server(config)
        server.run()

    async def _get_builtin_prompts(self) -> JSONResponse:
        """
        For internal usage: /v1/models/prompts
        获取内置的模型提示词模板
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_prompts()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_builtin_families(self) -> JSONResponse:
        """
        For internal usage: /v1/models/families
        获取内置的模型家族列表
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_families()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_devices_count(self) -> JSONResponse:
        """
        For internal usage: /v1/cluster/devices
        获取 CUDA 加速卡数量
        """
        try:
            data = await (await self._get_supervisor_ref()).get_devices_count()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_status(self) -> JSONResponse:
        """
        返回 worker 状态
        """
        try:
            data = await (await self._get_supervisor_ref()).get_status()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        
    async def get_address(self) -> JSONResponse:
        return JSONResponse(content=self._supervisor_address)
        
def run(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    auth_config_file: Optional[str] = None,
):
    logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
    try:
        api = RESTfulAPI(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            auth_config_file=auth_config_file,
        )
        api.serve(logging_conf=logging_conf)
    except SystemExit:
        logger.warning("Failed to create socket with port %d", port)
        # compare the reference to differentiate between the cases where the user specify the
        # default port and the user does not specify the port.
        if port is XINFERENCE_DEFAULT_ENDPOINT_PORT:
            port = get_next_port()
            logger.info(f"Found available port: {port}")
            logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
            api = RESTfulAPI(
                supervisor_address=supervisor_address,
                host=host,
                port=port,
                auth_config_file=auth_config_file,
            )
            api.serve(logging_conf=logging_conf)
        else:
            raise