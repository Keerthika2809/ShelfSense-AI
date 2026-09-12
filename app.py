import streamlit as st
import textwrap
import json
import os
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
from PIL import Image
from rag_engine import ShelfSenseRAG
from gemini_engine import ShelfSenseGemini

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ShelfSense AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ ShelfSense AI")

st.markdown(
    "### Visual Object Intelligence, RAG & Smart Organization Assistant"
)

st.write(
    "See objects. Understand them. Ask questions. "
    "Get intelligent organization suggestions."
)

st.divider()

# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


@st.cache_resource
def load_rag():
    return ShelfSenseRAG()


rag = load_rag()
@st.cache_resource
def load_gemini():
    return ShelfSenseGemini()


gemini = load_gemini()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🧭 ShelfSense AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🔍 Analyze",
        "🧠 Object Intelligence",
        "✨ Smart Arrange",
        "💬 AI Assistant",
        "🧑‍🏫 Teach ShelfSense",
        "📊 Dashboard"
    ]
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = []

if "image" not in st.session_state:
    st.session_state.image = None

if "results" not in st.session_state:
    st.session_state.results = None
# --------------------------------------------------
# ANALYSIS HISTORY
# --------------------------------------------------

HISTORY_FILE = "analysis_history.json"


def load_analysis_history():

    if not os.path.exists(HISTORY_FILE):

        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except:

        return []


def save_analysis_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


if "analysis_history" not in st.session_state:

    st.session_state.analysis_history = (
        load_analysis_history()
    )


# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.markdown(textwrap.dedent("""
    <style>
    .hero {
        padding: 50px 40px;
        border-radius: 25px;
        background: linear-gradient(135deg, #111827, #1e293b);
        color: white;
        margin-bottom: 30px;
    }
    .hero h1 {
        font-size: 52px;
        margin-bottom: 10px;
    }
    .hero p {
        font-size: 20px;
        color: #cbd5e1;
    }
    .card {
        padding: 25px;
        border-radius: 18px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        margin-bottom: 15px;
        min-height: 150px;
        color: #0f172a;
    }
    .card h3 {
        color: #0f172a !important;
        font-size: 21px;
        margin-bottom: 10px;
    }
    .card p {
        color: #475569 !important;
        font-size: 15px;
        line-height: 1.5;
    }
    </style>
    """), unsafe_allow_html=True)

    # ---------------- HERO ----------------

    st.markdown(textwrap.dedent("""
    <div class="hero">
        <div style="font-size:16px;">
            ✨ AI-POWERED VISUAL INTELLIGENCE
        </div>
        <h1>
            SEE IT. UNDERSTAND IT. ORGANIZE IT.
        </h1>
        <p>
            ShelfSense AI transforms ordinary images into
            intelligent, actionable insights using
            Computer Vision, RAG and Generative AI.
        </p>
    </div>
    """), unsafe_allow_html=True)

    # ---------------- FEATURES ----------------

    st.subheader("🚀 What ShelfSense AI Does")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>👁️ Visual Detection</h3>
            <p>
            Detect objects from images using YOLO
            computer vision and understand where
            they appear in the scene.
            </p>
        </div>
        """), unsafe_allow_html=True)

    with col2:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>🧠 Object Intelligence</h3>
            <p>
            Understand detected objects through
            categories, properties, uses,
            relationships and retrieved knowledge.
            </p>
        </div>
        """), unsafe_allow_html=True)

    with col3:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>💬 AI Assistant</h3>
            <p>
            Ask questions about your analyzed image
            and get contextual answers using
            RAG and Generative AI.
            </p>
        </div>
        """), unsafe_allow_html=True)

    with col4:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>✨ Smart Arrange</h3>
            <p>
            Get intelligent recommendations to
            organize objects based on space,
            category, accessibility and safety.
            </p>
        </div>
        """), unsafe_allow_html=True)

    # ---------------- WORKFLOW ----------------

    st.divider()

    st.subheader("🔄 How ShelfSense Works")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("01", "Capture")
    c2.metric("02", "Detect")
    c3.metric("03", "Understand")
    c4.metric("04", "Interact")
    c5.metric("05", "Recommend")

    # ---------------- USE CASES ----------------

    st.divider()

    st.subheader("🌍 Designed for Real-World Spaces")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            "### 📚 Library\n"
            "Organize books and learning materials."
        )

    with b:
        st.markdown(
            "### 🖥️ Study Space\n"
            "Understand desks and study objects."
        )

    with c:
        st.markdown(
            "### 🛒 Retail\n"
            "Analyze products and shelf arrangements."
        )

    with d:
        st.markdown(
            "### 🏠 Everyday Spaces\n"
            "Understand rooms and household objects."
        )

    # ---------------- FINAL MESSAGE ----------------

    st.divider()

    st.success(
        "🚀 ShelfSense AI is more than an object detector — "
        "it connects Computer Vision, Knowledge Retrieval, "
        "Generative AI and Smart Organization into one system."
    )

# ==================================================
# ANALYZE
# ==================================================

elif page == "🔍 Analyze":

    st.header("🔍 Visual Analysis")

    st.write(
        "Upload any scene — library, room, desk, shop, office or outdoor environment."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.session_state.image = image

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        if st.button("🚀 Analyze Image"):

            with st.spinner("AI is analyzing the scene..."):

                results = model(image)
                st.session_state.results = results

                detected_objects = []

                for box in results[0].boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Ignore weak detections
                    if confidence < 0.45:
                        continue

                    object_name = model.names[class_id]

                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Object center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Object size
                    width = x2 - x1
                    height = y2 - y1

                    detected_objects.append(
                        {
                            "name": object_name,
                            "confidence": confidence,

                            # Position
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,

                            # Center
                            "center_x": center_x,
                            "center_y": center_y,

                            # Size
                            "width": width,
                            "height": height
                        }
                    )

                st.session_state.detected_objects = detected_objects
		# --------------------------------------------------
                # SAVE ANALYSIS TO HISTORY
                # --------------------------------------------------

                history_record = {

                    "timestamp":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "object_count":
                        len(detected_objects),

                    "objects":
                        [
                            obj["name"]
                            for obj in detected_objects
                        ]
                }

                st.session_state.analysis_history.append(
                    history_record
                )

                save_analysis_history(
                    st.session_state.analysis_history
                )

            st.success("Analysis completed!")

    # --------------------------------------------------
    # DISPLAY DETECTION
    # --------------------------------------------------

    if st.session_state.results is not None:

        st.subheader("👁️ Detected Scene")

        result_image = st.session_state.results[0].plot()

        st.image(
            result_image,
            caption="AI Object Detection",
            width="stretch"
        )

        st.subheader("🏷️ Detected Objects")

        if st.session_state.detected_objects:
            for obj in st.session_state.detected_objects:
                st.write(
                    f"**{obj['name']}** "
                    f"— Confidence: {obj['confidence']:.2f}"
                )
        else:
            st.warning("No objects detected.")


# ==================================================
# OBJECT INTELLIGENCE
# ==================================================

elif page == "🧠 Object Intelligence":

    st.header("🧠 Object Intelligence")

    st.write(
        "Turn detected objects into meaningful knowledge "
        "using ShelfSense RAG."
    )

    if not st.session_state.detected_objects:

        st.info(
            "🔍 Analyze an image first to explore "
            "Object Intelligence."
        )

    else:

        st.success("🧠 Knowledge Retrieval Active")

        for obj in st.session_state.detected_objects:

            object_name = obj["name"]

            with st.expander(
                f"🔎 {object_name.title()}",
                expanded=True
            ):

                st.write(f"### {object_name.title()}")

                st.write(
                    f"**Detection confidence:** "
                    f"{obj['confidence']:.2f}"
                )

                knowledge = rag.search(
                    object_name,
                    top_k=1
                )

                if knowledge:

                    info = knowledge[0]

                    st.write(f"**📂 Category:** {info['category']}")
                    st.write(f"**📖 About:** {info['description']}")
                    st.write(f"**🔧 Uses:** {info['uses']}")
                    st.write(
                        f"**🔗 Related objects:** "
                        f"{info['related_objects']}"
                    )
                    st.write(
                        f"**📐 Organization guidance:** "
                        f"{info['organization']}"
                    )

                    st.caption(
                        "📚 Source: ShelfSense Knowledge Base "
                        "via semantic retrieval"
                    )

                else:

                    st.warning("No relevant knowledge found.")


# ==================================================
# SMART ARRANGE
# ==================================================

elif page == "✨ Smart Arrange":

    st.header("✨ Smart Arrange")

    st.write(
        "ShelfSense analyzes the actual positions, spacing, "
        "size and grouping of detected objects."
    )

    if not st.session_state.detected_objects:

        st.info(
            "🔍 Analyze an image first to use Smart Arrange."
        )

    else:

        objects = st.session_state.detected_objects
        image = st.session_state.image.copy()

        st.success(
            f"📦 {len(objects)} objects available for arrangement analysis"
        )

        # --------------------------------------------------
        # ORGANIZATION GOAL
        # --------------------------------------------------

        goal = st.selectbox(
            "🎯 What should ShelfSense optimize?",
            [
                "Neat and balanced",
                "Group similar objects",
                "Save space",
                "Easy accessibility",
                "Safety-oriented"
            ]
        )

        if st.button("✨ Generate Smart Arrangement"):

            with st.spinner(
                "🧠 Analyzing object positions and relationships..."
            ):

                # ==================================================
                # BASIC IMAGE INFORMATION
                # ==================================================

                image_width, image_height = image.size

                image_area = (
                    image_width *
                    image_height
                )

                # ==================================================
                # COPY OBJECT DATA
                # ==================================================

                data = []

                for index, obj in enumerate(objects):

                    x1 = float(obj["x1"])
                    y1 = float(obj["y1"])
                    x2 = float(obj["x2"])
                    y2 = float(obj["y2"])

                    width = max(
                        1.0,
                        x2 - x1
                    )

                    height = max(
                        1.0,
                        y2 - y1
                    )

                    center_x = (
                        x1 + x2
                    ) / 2

                    center_y = (
                        y1 + y2
                    ) / 2

                    area = (
                        width *
                        height
                    )

                    data.append(
                        {
                            "index": index,
                            "name": obj["name"],
                            "confidence": obj["confidence"],
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": width,
                            "height": height,
                            "center_x": center_x,
                            "center_y": center_y,
                            "area": area,
                            "nx": center_x / image_width,
                            "ny": center_y / image_height,
                            "nw": width / image_width,
                            "nh": height / image_height
                        }
                    )

                # ==================================================
                # HELPER FUNCTIONS
                # ==================================================

                def intersection_over_union(a, b):

                    left = max(
                        a["x1"],
                        b["x1"]
                    )

                    top = max(
                        a["y1"],
                        b["y1"]
                    )

                    right = min(
                        a["x2"],
                        b["x2"]
                    )

                    bottom = min(
                        a["y2"],
                        b["y2"]
                    )

                    intersection_width = max(
                        0,
                        right - left
                    )

                    intersection_height = max(
                        0,
                        bottom - top
                    )

                    intersection = (
                        intersection_width *
                        intersection_height
                    )

                    area_a = (
                        a["width"] *
                        a["height"]
                    )

                    area_b = (
                        b["width"] *
                        b["height"]
                    )

                    union = (
                        area_a +
                        area_b -
                        intersection
                    )

                    if union <= 0:
                        return 0

                    return (
                        intersection /
                        union
                    )


                def center_distance(a, b):

                    dx = (
                        a["center_x"] -
                        b["center_x"]
                    )

                    dy = (
                        a["center_y"] -
                        b["center_y"]
                    )

                    return (
                        dx * dx +
                        dy * dy
                    ) ** 0.5


                def normalize_name(name):

                    return (
                        name
                        .strip()
                        .lower()
                    )


                # ==================================================
                # DETECT VISUAL ROWS DYNAMICALLY
                # ==================================================

                sorted_objects = sorted(
                    data,
                    key=lambda item: item["center_y"]
                )

                rows = []

                average_height = sum(
                    item["height"]
                    for item in data
                ) / max(
                    1,
                    len(data)
                )

                row_tolerance = max(
                    25,
                    average_height * 0.55
                )

                for item in sorted_objects:

                    placed = False

                    for row in rows:

                        row_center = sum(
                            obj["center_y"]
                            for obj in row
                        ) / len(row)

                        if abs(
                            item["center_y"] -
                            row_center
                        ) <= row_tolerance:

                            row.append(item)
                            placed = True
                            break

                    if not placed:

                        rows.append(
                            [item]
                        )

                # Sort objects inside each visual row
                for row in rows:

                    row.sort(
                        key=lambda item:
                        item["center_x"]
                    )

                # ==================================================
                # ASSIGN ROW NUMBERS
                # ==================================================

                for row_number, row in enumerate(rows):

                    for item in row:

                        item["row"] = row_number


                # ==================================================
                # FIND SAME-OBJECT GROUPS
                # ==================================================

                groups = {}

                for item in data:

                    key = normalize_name(
                        item["name"]
                    )

                    if key not in groups:

                        groups[key] = []

                    groups[key].append(item)


                # ==================================================
                # FIND SPATIAL PROBLEMS
                # ==================================================

                problems = []

                for i in range(
                    len(data)
                ):

                    for j in range(
                        i + 1,
                        len(data)
                    ):

                        first = data[i]
                        second = data[j]

                        iou = intersection_over_union(
                            first,
                            second
                        )

                        distance = center_distance(
                            first,
                            second
                        )

                        min_dimension = min(
                            first["width"],
                            first["height"],
                            second["width"],
                            second["height"]
                        )

                        close_threshold = max(
                            30,
                            min_dimension * 0.45
                        )

                        same_type = (
                            normalize_name(
                                first["name"]
                            )
                            ==
                            normalize_name(
                                second["name"]
                            )
                        )

                        if iou > 0.08:

                            problems.append(
                                {
                                    "type": "overlap",
                                    "first": first,
                                    "second": second,
                                    "severity": iou
                                }
                            )

                        elif (
                            distance <
                            close_threshold
                            and
                            not same_type
                        ):

                            problems.append(
                                {
                                    "type": "crowding",
                                    "first": first,
                                    "second": second,
                                    "severity": (
                                        1 -
                                        distance /
                                        close_threshold
                                    )
                                }
                            )


                # ==================================================
                # DETERMINE GROUP ANCHORS
                # ==================================================

                group_anchors = {}

                for name, items in groups.items():

                    if len(items) <= 1:
                        continue

                    average_x = sum(
                        item["center_x"]
                        for item in items
                    ) / len(items)

                    average_y = sum(
                        item["center_y"]
                        for item in items
                    ) / len(items)

                    group_anchors[name] = {
                        "x": average_x,
                        "y": average_y,
                        "count": len(items)
                    }


                # ==================================================
                # FIND GOOD TARGET LOCATIONS
                # ==================================================
                #
                # IMPORTANT:
                # Targets are generated from REAL image geometry.
                # There is no fixed 3x3 grid.
                #
                # ==================================================

                candidate_targets = []

                for item in data:

                    # Current position itself
                    candidate_targets.append(
                        {
                            "x": item["center_x"],
                            "y": item["center_y"],
                            "source": "current"
                        }
                    )

                    # Spaces between neighboring objects
                    for other in data:

                        if other["index"] == item["index"]:
                            continue

                        if other["row"] != item["row"]:
                            continue

                        left = min(
                            item["center_x"],
                            other["center_x"]
                        )

                        right = max(
                            item["center_x"],
                            other["center_x"]
                        )

                        gap = (
                            right -
                            left
                        )

                        required_gap = (
                            item["width"] +
                            other["width"]
                        ) / 2

                        if gap > required_gap:

                            candidate_targets.append(
                                {
                                    "x": (
                                        left +
                                        right
                                    ) / 2,
                                    "y": (
                                        item["center_y"] +
                                        other["center_y"]
                                    ) / 2,
                                    "source": "gap"
                                }
                            )


                # ==================================================
                # ADD SAFE IMAGE-BOUNDARY TARGETS
                # ==================================================

                margin_x = image_width * 0.08
                margin_y = image_height * 0.08

                for item in data:

                    possible_positions = [

                        (
                            margin_x +
                            item["width"] / 2,
                            item["center_y"]
                        ),

                        (
                            image_width -
                            margin_x -
                            item["width"] / 2,
                            item["center_y"]
                        ),

                        (
                            item["center_x"],
                            margin_y +
                            item["height"] / 2
                        ),

                        (
                            item["center_x"],
                            image_height -
                            margin_y -
                            item["height"] / 2
                        )
                    ]

                    for px, py in possible_positions:

                        if (
                            px -
                            item["width"] / 2
                            >= 0
                            and
                            px +
                            item["width"] / 2
                            <= image_width
                            and
                            py -
                            item["height"] / 2
                            >= 0
                            and
                            py +
                            item["height"] / 2
                            <= image_height
                        ):

                            candidate_targets.append(
                                {
                                    "x": px,
                                    "y": py,
                                    "source": "boundary"
                                }
                            )


                # ==================================================
                # SCORE TARGETS
                # ==================================================

                recommendations = []

                occupied_targets = []

                for item in data:

                    best_target = None
                    best_score = -999999

                    current_name = normalize_name(
                        item["name"]
                    )

                    for target in candidate_targets:

                        tx = target["x"]
                        ty = target["y"]

                        # ------------------------------------------
                        # Keep target inside image
                        # ------------------------------------------

                        if (
                            tx -
                            item["width"] / 2
                            < 0
                            or
                            tx +
                            item["width"] / 2
                            > image_width
                            or
                            ty -
                            item["height"] / 2
                            < 0
                            or
                            ty +
                            item["height"] / 2
                            > image_height
                        ):

                            continue


                        # ------------------------------------------
                        # Movement cost
                        # ------------------------------------------

                        movement = (
                            (
                                tx -
                                item["center_x"]
                            ) ** 2
                            +
                            (
                                ty -
                                item["center_y"]
                            ) ** 2
                        ) ** 0.5

                        normalized_movement = (
                            movement /
                            max(
                                1,
                                (
                                    image_width +
                                    image_height
                                ) / 2
                            )
                        )


                        # ------------------------------------------
                        # Target collision
                        # ------------------------------------------

                        collision = 0

                        target_box = {
                            "x1":
                                tx -
                                item["width"] / 2,

                            "y1":
                                ty -
                                item["height"] / 2,

                            "x2":
                                tx +
                                item["width"] / 2,

                            "y2":
                                ty +
                                item["height"] / 2,

                            "width":
                                item["width"],

                            "height":
                                item["height"]
                        }

                        for other in data:

                            if (
                                other["index"]
                                ==
                                item["index"]
                            ):
                                continue

                            other_iou = intersection_over_union(
                                target_box,
                                other
                            )

                            collision += (
                                other_iou * 8
                            )


                        # ------------------------------------------
                        # Similar-object grouping
                        # ------------------------------------------

                        grouping_bonus = 0

                        if current_name in group_anchors:

                            anchor = group_anchors[
                                current_name
                            ]

                            anchor_distance = (
                                (
                                    tx -
                                    anchor["x"]
                                ) ** 2
                                +
                                (
                                    ty -
                                    anchor["y"]
                                ) ** 2
                            ) ** 0.5

                            normalized_anchor_distance = (
                                anchor_distance /
                                max(
                                    1,
                                    (
                                        image_width +
                                        image_height
                                    ) / 2
                                )
                            )

                            grouping_bonus = (
                                5 *
                                (
                                    1 -
                                    min(
                                        1,
                                        normalized_anchor_distance
                                    )
                                )
                            )


                        # ------------------------------------------
                        # Row preservation
                        # ------------------------------------------

                        row_penalty = 0

                        if item["row"] >= 0:

                            row_center = sum(
                                obj["center_y"]
                                for obj in rows[
                                    item["row"]
                                ]
                            ) / len(
                                rows[
                                    item["row"]
                                ]
                            )

                            row_difference = abs(
                                ty -
                                row_center
                            )

                            row_penalty = (
                                row_difference /
                                max(
                                    1,
                                    image_height
                                )
                            ) * 3


                        # ------------------------------------------
                        # Goal-specific score
                        # ------------------------------------------

                        goal_bonus = 0

                        if goal == "Neat and balanced":

                            distance_from_center = abs(
                                tx -
                                image_width / 2
                            )

                            balance = (
                                1 -
                                min(
                                    1,
                                    distance_from_center /
                                    (
                                        image_width / 2
                                    )
                                )
                            )

                            goal_bonus = (
                                balance * 1.5
                            )


                        elif goal == "Group similar objects":

                            goal_bonus = (
                                grouping_bonus *
                                1.8
                            )


                        elif goal == "Save space":

                            nearest_distance = (
                                image_width +
                                image_height
                            )

                            for other in data:

                                if (
                                    other["index"]
                                    ==
                                    item["index"]
                                ):
                                    continue

                                d = (
                                    (
                                        tx -
                                        other["center_x"]
                                    ) ** 2
                                    +
                                    (
                                        ty -
                                        other["center_y"]
                                    ) ** 2
                                ) ** 0.5

                                if d < nearest_distance:
                                    nearest_distance = d

                            compactness = (
                                1 -
                                min(
                                    1,
                                    nearest_distance /
                                    max(
                                        image_width,
                                        image_height
                                    )
                                )
                            )

                            goal_bonus = (
                                compactness * 2
                            )


                        elif goal == "Easy accessibility":

                            center_preference = (
                                1 -
                                abs(
                                    tx -
                                    image_width / 2
                                )
                                /
                                max(
                                    1,
                                    image_width / 2
                                )
                            )

                            vertical_preference = (
                                1 -
                                abs(
                                    ty -
                                    image_height / 2
                                )
                                /
                                max(
                                    1,
                                    image_height / 2
                                )
                            )

                            goal_bonus = (
                                (
                                    center_preference +
                                    vertical_preference
                                ) / 2
                            ) * 2


                        elif goal == "Safety-oriented":

                            # Keep objects away from image edges
                            # and away from detected crowding.

                            edge_distance = min(
                                tx,
                                image_width - tx,
                                ty,
                                image_height - ty
                            )

                            edge_score = min(
                                1,
                                edge_distance /
                                max(
                                    1,
                                    min(
                                        image_width,
                                        image_height
                                    ) * 0.15
                                )
                            )

                            goal_bonus = (
                                edge_score * 2
                            )


                        # ------------------------------------------
                        # Final target score
                        # ------------------------------------------

                        score = (

                            grouping_bonus

                            + goal_bonus

                            - (
                                normalized_movement *
                                5
                            )

                            - collision

                            - row_penalty
                        )


                        # Avoid selecting occupied target
                        for occupied in occupied_targets:

                            occupied_distance = (
                                (
                                    tx -
                                    occupied["x"]
                                ) ** 2
                                +
                                (
                                    ty -
                                    occupied["y"]
                                ) ** 2
                            ) ** 0.5

                            if occupied_distance < (
                                item["width"] *
                                0.75
                            ):

                                score -= 5


                        if score > best_score:

                            best_score = score

                            best_target = {
                                "x": tx,
                                "y": ty,
                                "score": score
                            }


                    # ==================================================
                    # DECIDE KEEP OR MOVE
                    # ==================================================

                    if best_target is None:

                        recommendations.append(
                            {
                                "index":
                                    item["index"],

                                "name":
                                    item["name"],

                                "action":
                                    "KEEP",

                                "reason":
                                    "No clearly better position was found.",

                                "current":
                                    (
                                        item["center_x"],
                                        item["center_y"]
                                    ),

                                "target":
                                    (
                                        item["center_x"],
                                        item["center_y"]
                                    ),

                                "score":
                                    0
                            }
                        )

                        continue


                    target_x = best_target["x"]
                    target_y = best_target["y"]

                    movement = (
                        (
                            target_x -
                            item["center_x"]
                        ) ** 2
                        +
                        (
                            target_y -
                            item["center_y"]
                        ) ** 2
                    ) ** 0.5


                    # Minimum movement threshold
                    movement_threshold = max(
                        30,
                        min(
                            item["width"],
                            item["height"]
                        ) * 0.55
                    )


                    # ==================================================
                    # LOOK FOR REAL PROBLEMS
                    # ==================================================

                    has_problem = False

                    for problem in problems:

                        if (
                            problem["first"]["index"]
                            ==
                            item["index"]
                            or
                            problem["second"]["index"]
                            ==
                            item["index"]
                        ):

                            has_problem = True
                            break


                    # Only move when improvement is meaningful
                    should_move = (
                        movement >
                        movement_threshold
                        and
                        (
                            has_problem
                            or
                            goal == "Group similar objects"
                        )
                    )


                    if should_move:

                        action = "MOVE"

                        reason = (
                            "Move this object to create "
                            "better spacing and organization."
                        )

                        occupied_targets.append(
                            {
                                "x": target_x,
                                "y": target_y
                            }
                        )

                    else:

                        action = "KEEP"

                        target_x = item["center_x"]
                        target_y = item["center_y"]

                        reason = (
                            "Current position is already "
                            "reasonable."
                        )


                    recommendations.append(
                        {
                            "index":
                                item["index"],

                            "name":
                                item["name"],

                            "action":
                                action,

                            "reason":
                                reason,

                            "current":
                                (
                                    item["center_x"],
                                    item["center_y"]
                                ),

                            "target":
                                (
                                    target_x,
                                    target_y
                                ),

                            "score":
                                best_target["score"]
                        }
                    )


                # ==================================================
                # CREATE VISUAL ARRANGEMENT IMAGE
                # ==================================================

                from PIL import ImageDraw, ImageFont

                draw = ImageDraw.Draw(
                    image,
                    "RGBA"
                )

                try:

                    font = ImageFont.truetype(
                        "arial.ttf",
                        18
                    )

                except:

                    font = ImageFont.load_default()


                # --------------------------------------------------
                # DRAW CURRENT OBJECT BOXES
                # --------------------------------------------------

                for item in data:

                    draw.rectangle(
                        [
                            item["x1"],
                            item["y1"],
                            item["x2"],
                            item["y2"]
                        ],
                        outline=(
                            255,
                            255,
                            255,
                            180
                        ),
                        width=2
                    )

                    label = (
                        f"{item['name']} "
                        f"{item['confidence']:.0%}"
                    )

                    draw.text(
                        (
                            item["x1"],
                            max(
                                0,
                                item["y1"] - 22
                            )
                        ),
                        label,
                        fill=(
                            255,
                            255,
                            255,
                            255
                        ),
                        font=font
                    )


                # --------------------------------------------------
                # DRAW RECOMMENDATIONS
                # --------------------------------------------------

                move_count = 0

                for rec in recommendations:

                    if rec["action"] != "MOVE":
                        continue

                    move_count += 1

                    current_x, current_y = (
                        rec["current"]
                    )

                    target_x, target_y = (
                        rec["target"]
                    )

                    # Target box size
                    item = data[
                        rec["index"]
                    ]

                    target_width = item[
                        "width"
                    ]

                    target_height = item[
                        "height"
                    ]

                    # ----------------------------------------------
                    # Target box
                    # ----------------------------------------------

                    draw.rectangle(
                        [
                            target_x -
                            target_width / 2,

                            target_y -
                            target_height / 2,

                            target_x +
                            target_width / 2,

                            target_y +
                            target_height / 2
                        ],
                        outline=(
                            0,
                            255,
                            120,
                            230
                        ),
                        width=4
                    )

                    # ----------------------------------------------
                    # Arrow
                    # ----------------------------------------------

                    draw.line(
                        [
                            current_x,
                            current_y,
                            target_x,
                            target_y
                        ],
                        fill=(
                            255,
                            80,
                            80,
                            240
                        ),
                        width=5
                    )

                    # ----------------------------------------------
                    # Target marker
                    # ----------------------------------------------

                    marker_size = 8

                    draw.ellipse(
                        [
                            target_x -
                            marker_size,

                            target_y -
                            marker_size,

                            target_x +
                            marker_size,

                            target_y +
                            marker_size
                        ],
                        fill=(
                            0,
                            255,
                            120,
                            230
                        )
                    )

                    # ----------------------------------------------
                    # MOVE label
                    # ----------------------------------------------

                    label = (
                        f"MOVE: {rec['name']}"
                    )

                    draw.text(
                        (
                            target_x + 10,
                            target_y - 10
                        ),
                        label,
                        fill=(
                            0,
                            255,
                            120,
                            255
                        ),
                        font=font
                    )


                # ==================================================
                # DISPLAY IMAGE
                # ==================================================

                st.subheader(
                    "🖼️ Recommended Arrangement"
                )

                st.image(
                    image,
                    use_container_width=True
                )


                # ==================================================
                # TEXT RECOMMENDATIONS
                # ==================================================

                st.subheader(
                    "📋 Arrangement Suggestions"
                )

                for rec in recommendations:

                    if rec["action"] == "MOVE":

                        current_x, current_y = (
                            rec["current"]
                        )

                        target_x, target_y = (
                            rec["target"]
                        )

                        dx = (
                            target_x -
                            current_x
                        )

                        dy = (
                            target_y -
                            current_y
                        )

                        if abs(dx) > abs(dy):

                            if dx > 0:

                                direction = "right"

                            else:

                                direction = "left"

                        else:

                            if dy > 0:

                                direction = "down"

                            else:

                                direction = "up"


                        st.warning(
                            f"🔄 **MOVE {rec['name']} → {direction}**\n\n"
                            f"{rec['reason']}"
                        )

                    else:

                        st.success(
                            f"✅ **KEEP {rec['name']}** — "
                            f"{rec['reason']}"
                        )


                # ==================================================
                # SUMMARY
                # ==================================================

                kept = sum(
                    1
                    for rec in recommendations
                    if rec["action"] == "KEEP"
                )

                moved = sum(
                    1
                    for rec in recommendations
                    if rec["action"] == "MOVE"
                )


                st.info(
                    f"📊 **Arrangement Summary:** "
                    f"{moved} object(s) can be improved, "
                    f"{kept} object(s) are already reasonably placed."
                )


                # ==================================================
                # GEMINI EXPLANATION
                # ==================================================

                st.subheader(
                    "🤖 ShelfSense AI Explanation"
                )

                arrangement_data = []

                for rec in recommendations:

                    arrangement_data.append(
                        {
                            "object":
                                rec["name"],

                            "action":
                                rec["action"],

                            "reason":
                                rec["reason"],

                            "current_position":
                                {
                                    "x":
                                        round(
                                            rec["current"][0],
                                            1
                                        ),

                                    "y":
                                        round(
                                            rec["current"][1],
                                            1
                                        )
                                },

                            "recommended_position":
                                {
                                    "x":
                                        round(
                                            rec["target"][0],
                                            1
                                        ),

                                    "y":
                                        round(
                                            rec["target"][1],
                                            1
                                        )
                                }
                        }
                    )


                prompt = f"""
You are ShelfSense AI.

You are explaining an arrangement decision that has
ALREADY been calculated by a Python computer-vision
engine.

IMPORTANT:
Do NOT create new objects.
Do NOT invent positions.
Do NOT contradict the Python decisions.
Do NOT claim that an object is on a specific shelf
unless the visual data supports it.

Detected objects:
{[item["name"] for item in data]}

Organization goal:
{goal}

Python arrangement decisions:
{arrangement_data}

Write a clear user-friendly explanation.

Rules:

1. Mention only detected objects.
2. Explain why objects marked MOVE should be moved.
3. Explain why objects marked KEEP can stay.
4. Do not invent additional movements.
5. Do not invent shelf levels.
6. Do not claim exact physical safety guarantees.
7. If the goal is safety-oriented, explain that
   the recommendation is based only on visible image
   information.
8. Keep the explanation practical.
9. Use short bullet points.
10. Finish with one concise overall recommendation.
"""

                try:

                    response = (
                        gemini.model.generate_content(
                            prompt
                        )
                    )

                    st.write(
                        response.text
                    )

                except Exception as e:

                    st.warning(
                        "Gemini explanation could not be generated."
                    )

                    st.write(
                        "The Python arrangement analysis "
                        "and visual recommendations are still available."
                    )
      
# ==================================================
# AI ASSISTANT
# ==================================================

elif page == "💬 AI Assistant":

    st.header("💬 ShelfSense AI Assistant")

    st.write(
        "Ask questions about the objects detected in your image."
    )

    if not st.session_state.detected_objects:

        st.info(
            "🔍 Analyze an image first to activate the AI Assistant."
        )

    else:

        detected = ", ".join(
            obj["name"]
            for obj in st.session_state.detected_objects
        )

        st.success("🧠 Scene context loaded")

        st.write(
            f"**Objects detected:** {detected}"
        )

        question = st.text_input(
            "💬 Ask ShelfSense about this scene..."
        )

        if st.button("🤖 Ask ShelfSense AI"):

            if question:

                with st.spinner(
                    "ShelfSense is thinking..."
                ):

                    # Get RAG knowledge
                    rag_knowledge = []

                    for obj in st.session_state.detected_objects:

                        knowledge = rag.search(
                            obj["name"],
                            top_k=1
                        )

                        if knowledge:
                            rag_knowledge.extend(
                                knowledge
                            )

                    # Remove duplicate knowledge
                    unique_knowledge = {}

                    for item in rag_knowledge:

                        object_name = (
                            item["object"]
                            .strip()
                            .lower()
                        )

                        unique_knowledge[
                            object_name
                        ] = item

                    rag_knowledge = list(
                        unique_knowledge.values()
                    )

                    # Ask Gemini
                    answer = gemini.ask(
                        question=question,
                        detected_objects=(
                            st.session_state.detected_objects
                        ),
                        rag_knowledge=rag_knowledge
                    )

                st.subheader("🤖 ShelfSense AI")

                st.write(answer)

            else:

                st.warning(
                    "Please enter a question."
                )

# ==================================================
# TEACH SHELFSENSE
# ==================================================

elif page == "🧑‍🏫 Teach ShelfSense":

    st.header("🧑‍🏫 Teach ShelfSense")

    st.write(
        """
        Add knowledge about an object that ShelfSense does not
        understand well. This information becomes part of the
        project's knowledge base.
        """
    )

    st.info(
        "💡 ShelfSense will remember what you teach it "
        "and use this knowledge in future RAG searches."
    )

    object_name = st.text_input(
        "🏷️ Object name"
    )

    category = st.text_input(
        "📂 Category"
    )

    description = st.text_area(
        "📝 Description"
    )

    uses = st.text_area(
        "🔧 Uses / important information"
    )

    related_objects = st.text_input(
        "🔗 Related objects",
        placeholder="Example: cup, plate, spoon"
    )

    organization = st.text_area(
        "📦 Organization advice",
        placeholder=(
            "Example: Keep this object near similar items "
            "and avoid placing fragile objects around it."
        )
    )


    if st.button(
        "🧠 Add to Knowledge Base"
    ):

        if not object_name.strip():

            st.warning(
                "⚠️ Please provide an object name."
            )

        elif not description.strip():

            st.warning(
                "⚠️ Please provide a description."
            )

        else:

            success, message = (
                rag.add_knowledge(

                    object_name=
                        object_name,

                    category=
                        category,

                    description=
                        description,

                    uses=
                        uses,

                    related_objects=
                        related_objects,

                    organization=
                        organization
                )
            )


            if success:

                st.success(
                    f"✅ {message}"
                )

                st.balloons()

                st.write(
                    "🧠 ShelfSense has learned this knowledge "
                    "and added it to the RAG knowledge base."
                )


 
# ==================================================
# CURRENT KNOWLEDGE
# ==================================================

    st.divider()

    with st.expander(
        f"📚 ShelfSense Knowledge Base — {len(rag.knowledge)} objects"
    ):

        st.write(
            "ShelfSense currently has knowledge about "
            f"**{len(rag.knowledge)} objects**."
        )

        search_object = st.text_input(
            "🔎 Search the knowledge base",
            placeholder="Type an object name..."
        )

        filtered_objects = rag.knowledge

        if search_object.strip():

            filtered_objects = [
                item
                for item in rag.knowledge
                if search_object.strip().lower()
                in item["object"].lower()
            ]

        if not filtered_objects:

            st.info(
                "No matching objects found."
            )

        else:

            for item in filtered_objects:

                with st.expander(
                    f"📦 {item['object']}"
                ):

                    st.write(
                        f"**Category:** {item['category']}"
                    )

                    st.write(
                        f"**Description:** "
                        f"{item['description']}"
                    )

                    st.write(
                        f"**Uses:** {item['uses']}"
                    )

                    st.write(
                        f"**Related objects:** "
                        f"{item['related_objects']}"
                    )

                    st.write(
                        f"**Organization:** "
                        f"{item['organization']}"
                    )


# ==================================================
# DASHBOARD
# ==================================================

elif page == "📊 Dashboard":

    st.header("📊 ShelfSense Dashboard")

    st.write(
        "Monitor your visual analyses, detected objects "
        "and scene intelligence."
    )

    history = st.session_state.analysis_history

    # --------------------------------------------------
    # CALCULATE STATISTICS
    # --------------------------------------------------

    total_images = len(history)

    total_objects_detected = sum(
        record["object_count"]
        for record in history
    )

    current_objects = len(
        st.session_state.detected_objects
    )

    # --------------------------------------------------
    # OBJECT FREQUENCY
    # --------------------------------------------------

    all_objects = []

    for record in history:

        all_objects.extend(
            record["objects"]
        )

    object_frequency = Counter(
        all_objects
    )

    # --------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------

    st.subheader("📈 ShelfSense Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📸 Images Analyzed",
            total_images
        )

    with col2:

        st.metric(
            "📦 Objects Detected",
            total_objects_detected
        )

    with col3:

        st.metric(
            "🔍 Current Objects",
            current_objects
        )

    with col4:

        st.metric(
            "🏆 Unique Objects",
            len(object_frequency)
        )

    # --------------------------------------------------
    # CURRENT SCENE
    # --------------------------------------------------

    st.divider()

    st.subheader("👁️ Current Scene")

    if current_objects == 0:

        st.info(
            "No current scene available. "
            "Go to Analyze and upload an image."
        )

    else:

        st.success(
            f"🧠 Current scene contains "
            f"{current_objects} detected object(s)."
        )

        scene_columns = st.columns(
            min(4, current_objects)
        )

        for index, obj in enumerate(
            st.session_state.detected_objects
        ):

            with scene_columns[
                index % len(scene_columns)
            ]:

                st.metric(
                    obj["name"].title(),
                    f"{obj['confidence']:.0%}"
                )

    # --------------------------------------------------
    # MOST FREQUENT OBJECTS
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "🏆 Most Frequently Detected Objects"
    )

    if not object_frequency:

        st.info(
            "Analyze more images to build "
            "object statistics."
        )

    else:

        top_objects = object_frequency.most_common(
            10
        )

        for rank, (object_name, count) in enumerate(
            top_objects,
            start=1
        ):

            st.write(
                f"**#{rank} {object_name.title()}** "
                f"— detected {count} time(s)"
            )

    # --------------------------------------------------
    # ANALYSIS HISTORY
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "🕒 Recent Analysis History"
    )

    if not history:

        st.info(
            "No analysis history yet."
        )

    else:

        # Show newest analyses first

        recent_history = list(
            reversed(history)
        )

        for index, record in enumerate(
            recent_history
        ):

            with st.expander(
                f"📸 Analysis {index + 1} — "
                f"{record['timestamp']}"
            ):

                st.write(
                    f"**Objects detected:** "
                    f"{record['object_count']}"
                )

                if record["objects"]:

                    st.write(
                        "**Detected objects:**"
                    )

                    for object_name in record[
                        "objects"
                    ]:

                        st.write(
                            f"• {object_name.title()}"
                        )

                else:

                    st.write(
                        "No objects detected."
                    )

    # --------------------------------------------------
    # DASHBOARD SUMMARY
    # --------------------------------------------------

    st.divider()

    st.success(
        "🚀 ShelfSense continuously builds an "
        "understanding of your analyzed scenes "
        "through computer vision, RAG intelligence "
        "and AI-powered recommendations."
    )

