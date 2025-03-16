from typing import List
import cv2
import numpy as np
from asgiref.sync import sync_to_async
from fastapi import Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
import io

from apps.licenseplates.models import Plate
from common.logic.image_processing import detect_license_plate, detect_license_plate_enhanced
from common.security import reusable_oauth2
from schemas.license_plate import LicensePlateBase, LicensePlateCreate

router = InferringRouter()


@cbv(router)
class LicensePlateView:

    @router.post(
        '/create-license-plate',
        dependencies=[Depends(reusable_oauth2)],
    )
    def create_license_plate(self, license_plate: LicensePlateCreate):
        """Create license plate for later check."""
        db_license_plate = Plate.objects.create(**license_plate.dict())
        return db_license_plate

    @router.post(
        '/read-license-plate',
        dependencies=[Depends(reusable_oauth2)],
    )
    async def read_license_plate(
        self,
        file: UploadFile,
        upscale: bool = True,
    ) -> List[LicensePlateBase]:
        """Read license plate with image."""
        contents = await file.read()

        nparr = np.fromstring(contents, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        list_read_plates = detect_license_plate(img, upscale=upscale)

        license_plate = await sync_to_async(list)(Plate.objects.filter(
            number__in=list_read_plates,
        ).values("number", "wanted"))
        if license_plate is None:
            raise HTTPException(
                status_code=404, detail='License plate not found')

        return license_plate

    @router.post(
        '/enhance-license-plate',
        dependencies=[Depends(reusable_oauth2)],
    )
    async def enhance_license_plate(
        self,
        file: UploadFile,
        upscale: bool = True,
    ):
        """Enhance license plate with image."""
        contents = await file.read()

        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)  # Use `np.frombuffer` instead of `np.fromstring`

        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        new_img = detect_license_plate_enhanced(img, upscale=upscale)


        # Encode the image back to JPEG format
        _, encoded_img = cv2.imencode('.jpg', new_img)

        # Convert to bytes
        img_bytes = io.BytesIO(encoded_img.tobytes())

        # Return as StreamingResponse
        return StreamingResponse(img_bytes, media_type="image/jpeg")
