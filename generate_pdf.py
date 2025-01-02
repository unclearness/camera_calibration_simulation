import cv2
from fpdf import FPDF


def generate_charucomarker_pdf(
    board, marker_size_mm, pdf_filename, dpi=72, place_center=True
):
    # Conversion factors
    mm_to_inch = 1 / 25.4
    pixels_per_mm = dpi * mm_to_inch

    if board.getChessboardSize()[0] > board.getChessboardSize()[1]:
        board_max_num = board.getChessboardSize()[0]
        ori_idx = 0
        orientation = "landscape"
    else:
        board_max_num = board.getChessboardSize()[1]
        ori_idx = 1
        orientation = "portrait"

    marker_org_len_pix = board.getSquareLength()

    possible_sizes = {
        "A4": (210, 297),
        "A3": (297, 420),
        "A2": (420, 594),
        "A1": (594, 841),
        "A0": (841, 1189),
    }
    page_size_mm = None
    page_size = None
    board_max_len_mm = board_max_num * marker_size_mm
    for size in possible_sizes:
        if possible_sizes[size][ori_idx] > board_max_len_mm:
            page_size = size
            page_size_mm = possible_sizes[size]
            break

    if page_size_mm is None:
        raise ValueError("Board is too large for any page size")

    # Calculate marker size in pixels
    marker_len_pix = int(max(marker_size_mm * pixels_per_mm, marker_org_len_pix))

    board_size_pix = (
        int(board.getChessboardSize()[0] * marker_len_pix),
        int(board.getChessboardSize()[1] * marker_len_pix),
    )
    board_img = board.generateImage(board_size_pix, marginSize=0, borderBits=1)
    # Create a new PDF
    pdf = FPDF(orientation=orientation, unit="mm", format=page_size)
    pdf.add_page()
    board_size_mm = (
        board.getChessboardSize()[0] * marker_size_mm,
        board.getChessboardSize()[1] * marker_size_mm,
    )
    if False:
        from PIL import Image

        pil_image = Image.fromarray(board_img)
        image_data = pil_image

    _, png_data = cv2.imencode(".png", board_img)
    image_data = png_data.tobytes()

    x_offset, y_offset = 0.0, 0.0
    if place_center:
        import fpdf

        x_offset = fpdf.enums.Align.C
        y_offset = (page_size_mm[1] - board_size_mm[1]) / 2
        y_offset /= 2  # TODO: I don't know why this is needed
    pdf.image(
        image_data,
        x=x_offset,
        y=y_offset,
        w=board_size_mm[0],
        h=board_size_mm[1],
        keep_aspect_ratio=True,
    )

    pdf.output(pdf_filename)


# Example usage
if __name__ == "__main__":
    # Create a sample ArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((11, 8), 12, 9, aruco_dict)
    generate_charucomarker_pdf(board, 12, "./data/charuco.pdf")
