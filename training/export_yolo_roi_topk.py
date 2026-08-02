from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "yolov8n_roi_320_30.onnx"
OUTPUT_PATH = ROOT / "yolov8n_roi_320_30_topk.onnx"
MODEL_INPUT_SHAPE = (1, 3, 32, 320)
MODEL_OUTPUT_SHAPE = [1, 5, 210]
TOPK_CANDIDATES = 16


def tensor_shape(value_info):
    """Return a static ONNX tensor shape as a list."""
    return [
        dimension.dim_value
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def main():
    model = onnx.load(str(SOURCE_PATH))
    onnx.checker.check_model(model)

    if len(model.graph.output) != 1:
        raise RuntimeError("expected one YOLO output")
    if tensor_shape(model.graph.input[0]) != list(MODEL_INPUT_SHAPE):
        raise RuntimeError(
            "unexpected YOLO input: %s" % tensor_shape(model.graph.input[0])
        )
    if tensor_shape(model.graph.output[0]) != MODEL_OUTPUT_SHAPE:
        raise RuntimeError(
            "unexpected YOLO output: %s" % tensor_shape(model.graph.output[0])
        )

    source_output = model.graph.output[0].name
    score_indices = "topk_score_indices"
    topk_count = "topk_count"
    expanded_shape = "topk_expanded_shape"
    topk_scores = "topk_scores"
    topk_indices = "topk_indices"
    expanded_indices = "topk_expanded_indices"
    final_output = "topk_detections"

    model.graph.initializer.extend(
        [
            # The single steel-ball class score is output channel 4.
            numpy_helper.from_array(
                np.asarray([4], dtype=np.int64), name=score_indices
            ),
            numpy_helper.from_array(
                np.asarray([TOPK_CANDIDATES], dtype=np.int64),
                name=topk_count,
            ),
            numpy_helper.from_array(
                np.asarray([1, 5, TOPK_CANDIDATES], dtype=np.int64),
                name=expanded_shape,
            ),
        ]
    )
    model.graph.node.extend(
        [
            helper.make_node(
                "Gather",
                [source_output, score_indices],
                ["topk_score_plane"],
                axis=1,
                name="SelectClassScores",
            ),
            helper.make_node(
                "TopK",
                ["topk_score_plane", topk_count],
                [topk_scores, topk_indices],
                axis=2,
                largest=1,
                sorted=1,
                name="SelectTopCandidates",
            ),
            helper.make_node(
                "Expand",
                [topk_indices, expanded_shape],
                [expanded_indices],
                name="ExpandCandidateIndices",
            ),
            helper.make_node(
                "GatherElements",
                [source_output, expanded_indices],
                [final_output],
                axis=2,
                name="GatherTopDetections",
            ),
        ]
    )

    # Expose only the compact TopK tensor to reduce board-side output handling.
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                final_output,
                TensorProto.FLOAT,
                [1, 5, TOPK_CANDIDATES],
            )
        ]
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(OUTPUT_PATH))

    # Numerically verify that the graph returns the same 16 candidates as NumPy.
    source_session = ort.InferenceSession(
        str(SOURCE_PATH), providers=["CPUExecutionProvider"]
    )
    topk_session = ort.InferenceSession(
        str(OUTPUT_PATH), providers=["CPUExecutionProvider"]
    )
    random_generator = np.random.default_rng(0)
    input_data = random_generator.random(MODEL_INPUT_SHAPE).astype(np.float32)
    input_name = source_session.get_inputs()[0].name
    source = source_session.run(None, {input_name: input_data})[0]
    topk = topk_session.run(None, {input_name: input_data})[0]

    expected_indices = np.argsort(source[:, 4, :], axis=1)[:, ::-1]
    expected_indices = expected_indices[:, :TOPK_CANDIDATES]
    expected = np.take_along_axis(
        source, expected_indices[:, None, :], axis=2
    )
    max_error = float(np.max(np.abs(expected - topk)))
    if topk.shape != (1, 5, TOPK_CANDIDATES) or max_error > 1e-5:
        raise RuntimeError(
            "TopK validation failed: shape=%s max_error=%g"
            % (topk.shape, max_error)
        )

    print("TopK ONNX generated:", OUTPUT_PATH)
    print("Output shape:", topk.shape)
    print("Maximum validation error:", max_error)


if __name__ == "__main__":
    main()
