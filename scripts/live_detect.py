import cv2
from src.features.color_hist import *
from src.features.structural import *
from src.features.print_quality import *
from src.features.template import *
from src.score import fuse_components
from src.model import verdict_for

reference = cv2.imread("samples/reference.jpg")
cap = cv2.VideoCapture(0)

print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_ref = cv2.resize(reference, (frame.shape[1], frame.shape[0]))

    try:
        color_score = compare_color_histograms(frame, resized_ref)
        ssim_score = compare_structural(frame, resized_ref)
        print_score = analyze_print_quality(frame)
        template_score = match_template(frame, resized_ref)

        components = {
            "template": {"score": template_score},
            "color": {"score": color_score},
            "ssim": {"score": ssim_score},
            "print_quality": {"score": print_score},
        }

        fused_score, _ = fuse_components(components)
        verdict = verdict_for(fused_score)

        text = f"{verdict} ({fused_score:.1f})"
        color = (0, 255, 0) if "GENUINE" in verdict else (0, 0, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)[:40]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Live Product Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
