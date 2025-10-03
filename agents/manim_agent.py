import logging
import time
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base import BaseAgent
from agents.concept_interpreter import ConceptAnalysis, SubConcept
from agents.manim_models import (
    ScenePlan, SceneAction, ManimSceneCode, RenderResult,
    AnimationResult, AnimationConfig, AnimationMetadata
)
from rendering.manim_renderer import ManimRenderer


class ManimAgent(BaseAgent):
    """
    Manim Agent: Transforms structured concept analysis into visual animations
    using scene planning and Manim code generation with <manim> tag extraction.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        output_dir: Path,
        config: Optional[AnimationConfig] = None,
        reasoning_tokens: Optional[float] = None
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, reasoning_tokens=reasoning_tokens)
        self.output_dir = Path(output_dir)
        self.config = config or AnimationConfig()

        # Initialize renderer
        self.renderer = ManimRenderer(
            output_dir=self.output_dir / "scenes",
            quality=self.config.quality,
            background_color=self.config.background_color,
            timeout=self.config.render_timeout,
            max_retries=self.config.max_retries_per_scene
        )

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_codes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "animations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scenes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_plans").mkdir(parents=True, exist_ok=True)

    SCENE_PLANNING_PROMPT = """You are a Manim Scene Planning Agent for an educational STEM animation system.

**TASK**: Create detailed scene plans for animating STEM concepts using Manim (Mathematical Animation Engine).

**INPUT CONCEPT ANALYSIS**:
{concept_analysis}

**ANIMATION GUIDELINES**:

1. **Scene Structure**:
   - Create 1-2 scenes per sub-concept (maximum 8 scenes total)
   - Each scene should be 15-45 seconds long
   - Build scenes logically following sub-concept dependencies
   - Start with foundations, progressively add complexity

2. **Visual Design**:
   - Use clear, educational visual style (dark background, bright elements)
   - Include mathematical notation, equations, diagrams
   - Show relationships and transformations visually
   - Use color coding for consistency (e.g., blue for known, green for new, red for important)

3. **Animation Types**:
   - Write/Create: Text, equations appearing
   - Transform/Replace: Mathematical transformations
   - Fade In/Out: Elements appearing/disappearing
   - Move/Highlight: Drawing attention to key elements
   - Grow/Shrink: Emphasizing scale or importance

4. **Educational Flow**:
   - Start with context/overview
   - Introduce new elements step-by-step
   - Show relationships and connections
   - End with key takeaways or summaries

**OUTPUT FORMAT**:
Return ONLY valid JSON matching this exact structure:
{{
    "scene_plans": [
        {{
            "id": "string",
            "title": "string",
            "description": "string",
            "sub_concept_id": "string",
            "actions": [
                {{
                    "action_type": "string",
                    "element_type": "string",
                    "description": "string",
                    "target": "string",
                    "duration": number,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["string"]
        }}
    ]
}}

**EXAMPLE** for Bayes' Theorem:
{{
    "scene_plans": [
        {{
            "id": "introduction",
            "title": "Bayes' Theorem Introduction",
            "description": "Introduce Bayes' theorem and its components with visual setup",
            "sub_concept_id": "prior_probability",
            "actions": [
                {{
                    "action_type": "fade_in",
                    "element_type": "text",
                    "description": "Display title 'Bayes' Theorem'",
                    "target": "title_text",
                    "duration": 2.0,
                    "parameters": {{"text": "Bayes' Theorem", "color": "#FFFFFF"}}
                }},
                {{
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Write the Bayes' theorem equation P(A|B) = P(B|A)P(A)/P(B)",
                    "target": "bayes_equation",
                    "duration": 4.0,
                    "parameters": {{"equation": "P(A|B) = \\\\frac{P(B|A)P(A)}{P(B)}", "color": "#00FF00"}}
                }}
            ],
            "scene_dependencies": []
        }}
    ]
}}

Generate scene plans that will create clear, educational animations for the given concept."""

    CODE_GENERATION_PROMPT = """You are a Manim Code Generation Agent for creating **very simple, 2D educational STEM animations**.

**TASK**: Generate complete, executable Manim code for a single animation scene using **Manim Community Edition v0.19**.

**SCENE PLAN**:
{scene_plan}

**SIMPLE 2D-ONLY MODE (STRICT)**

0. **Hard Limits (Do Not Violate)**
   - **2D only**: No 3D classes, no 3D cameras, no surfaces, no axes3D, no `ThreeDScene`.
   - **One scene, one class**, must inherit from `Scene`.
   - **One file** with exactly one class named **{class_name}**.
   - **Single `construct(self)` method** only.
   - **No updaters** of any kind (no `add_updater`, no `always_redraw`).
   - **No ValueTracker/DecimalNumber**; keep logic static and stepwise.
   - **No config edits, no camera/frame changes, no run_time tweaks** except via `self.wait()`.
   - **Keep it small**: Max **35 lines** inside `construct()` (excluding comments/blank lines).

1. **Imports**
   - Always start with:
     `from manim import *`
   - Optionally:
     `import numpy as np` **only if actually used**.
   - Do **not** import anything else.

2. **Text & Equations**
   - Regular text: `Text()`
   - Math: `MathTex()` (LaTeX in math mode)
   - Plain LaTeX (non-math): `Tex()`
   - Examples:
     - `title = Text("Gradient Descent", font_size=48)`
     - `eq = MathTex(r"\\nabla f(x) = \\frac{{df}}{{dx}}")`

3. **Allowed Mobjects (2D only)**
   - `Dot`, `Line`, `Arrow`, `Vector`, `Circle`, `Square`, `Rectangle`, `Triangle`
   - `NumberPlane`, `Axes` (2D only)
   - `Brace`, `SurroundingRectangle`
   - `Text`, `MathTex`, `Tex`
   - `VGroup`
   - **Do not** use SVGMobject, Parametric curves, graphs requiring functions beyond the basics, or any 3D variants.

4. **Allowed Animations (Keep It Basic)**
   - `Write(mobj)` for text/equations
   - `Create(mobj)` for geometric shapes/axes
   - `FadeIn(mobj)`, `FadeOut(mobj)`
   - `Transform(m1, m2)`, `ReplacementTransform(m1, m2)`
   - `Indicate(mobj)`
   - `self.play(...)` to run animations
   - `self.wait(t)` for pacing
   - **No** `ApplyMethod`, **use** `.animate` **only** for simple position/opacity/scale changes (e.g., `mobj.animate.shift(RIGHT)`)

5. **Positioning (Simple & Explicit)**
   - `to_edge(UP/DOWN/LEFT/RIGHT)`
   - `shift(UP/DOWN/LEFT/RIGHT * amount)`
   - `next_to(other, direction)`
   - `move_to(ORIGIN)` / leave centered by default
   - Avoid complex coordinate math; prefer simple shifts.

6. **Colors (Built-ins Only)**
   - Use constants: `WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK` - DO NOT USE CYAN EVER
   - Example: `Text("Hello", color=BLUE)`

7. **Flow (Minimal & Clear)**
   - Start with a short title (2–3 seconds)
   - Introduce content step-by-step (never all at once)
   - Add a brief `self.wait(1)` between logical steps
   - End with `self.wait(2)` to hold the final state
   - **Target total duration**: {target_duration} seconds (approximate; don’t over-animate)

8. **Forbidden (to reduce breakage)**
   - **No**: TexMobject, TextMobject, ShowCreation, UpdateFromFunc, apply_function, complex updaters
   - **No**: camera movements, zooms, rotating the frame, backgrounds changes, scene-wide transforms
   - **No**: randomness, seeds, external assets (SVGS, images, sounds), 3D of any kind
   - **No**: long code, clever hacks, or experimental APIs

9. **Style & Robustness**
   - Clear variable names: `title`, `equation`, `plane`, `arrow_x`
   - Keep geometry simple (few objects, few transformations)
   - Prefer `Create`/`Write` + simple `.animate.shift(...)`
   - Ensure every created mobject is added via an animation or `self.add(...)` before transforming it
   - Do **not** transform or reference mobjects after they’ve been `FadeOut` unless re-created
   
**REMINDERS**
- Only 2D classes & objects used; inherits from `Scene`
- Single class named `{class_name}`; one `construct` method
- Only allowed imports; optional numpy only if used
- Only `Text`, `Tex`, `MathTex`, basic 2D shapes, `NumberPlane`/`Axes`
- Only `Write`, `Create`, `FadeIn/Out`, `Transform`, `ReplacementTransform`, `Indicate`, and simple `.animate`
- No updaters/ValueTracker/camera edits/external assets/3D/experimental APIs
- Stepwise reveal with short waits; ends with `self.wait(3)`
- Please make slow animation for the narrator to have enough time to explain the concept, do not make animation and scene switching too fast
- Wait a bit between each step of the animation
- Make sure that all the elements are place correctly, not overlapping or being overflown on the screen
- Code is clean, minimal, and executable in Manim CE v0.19
- For text/animated text please be careful with the position where we put it, avoid having the text overlapping with other elements on the screen
- DO NOT INCLUDE BACKTICKS (```) IN YOUR CODE, EVER!

**OUTPUT FORMAT (MANDATORY)**
<manim>
from manim import *
# Manim Community v0.19 — Simple 2D Scene

class {class_name}(Scene):
    def construct(self):
        # Title
        title = Text("Your Title", font_size=48)
        self.play(Write(title))
        self.wait(1)

        # Your simple 2D animation code here (use only allowed mobjects/animations)
        # Example scaffold:
        # eq = MathTex(r"y = mx + b").next_to(title, DOWN)
        # self.play(Write(eq)); self.wait(1)
        # plane = NumberPlane()
        # self.play(Create(plane)); self.wait(1)
        # dot = Dot(ORIGIN, color=RED)
        # self.play(FadeIn(dot)); self.wait(1)
        # self.play(dot.animate.shift(RIGHT*2 + UP))
        # self.wait(1)

        self.wait(2)
</manim>
"""

    def execute(self, concept_analysis: ConceptAnalysis) -> AnimationResult:

        start_time = time.time()
        self.logger.info(f"Starting animation generation for: {concept_analysis.main_concept}")

        try:
            # Step 1: Generate scene plans
            scene_plans = self._generate_scene_plans(concept_analysis)
            self.logger.info(f"Generated {len(scene_plans)} scene plans")

            # Step 2: Generate Manim code for each scene
            scene_codes = self._generate_scene_codes(scene_plans)
            self.logger.info(f"Generated code for {len(scene_codes)} scenes")

            # Step 3: Render each scene
            render_results = self._render_scenes(scene_codes)
            successful_renders = [r for r in render_results if r.success]
            self.logger.info(f"Successfully rendered {len(successful_renders)}/{len(render_results)} scenes")

            # Step 4: Concatenate scenes into single animation
            if successful_renders:
                silent_animation_path = self._concatenate_scenes(successful_renders)
            else:
                silent_animation_path = None

            # Calculate timing
            generation_time = time.time() - start_time
            total_render_time = sum(r.render_time or 0 for r in render_results)

            # Create result
            result = AnimationResult(
                success=len(successful_renders) > 0,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                total_duration=sum(r.duration for r in successful_renders if r.duration),
                scene_count=len(scene_plans),
                silent_animation_path=str(silent_animation_path) if silent_animation_path else None,
                scene_plan=scene_plans,
                scene_codes=scene_codes,
                render_results=render_results,
                generation_time=generation_time,
                total_render_time=total_render_time,
                models_used={"reasoning": self.model},
                token_usage=self.get_token_usage()
            )

            self.logger.info(f"Animation generation completed in {generation_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Animation generation failed: {e}")
            return AnimationResult(
                success=False,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                scene_count=0,
                error_message=str(e),
                scene_plan=[],
                scene_codes=[],
                render_results=[],
                generation_time=time.time() - start_time,
                models_used={"reasoning": self.model},
                token_usage=self.get_token_usage()
            )

    def _generate_scene_plans(self, concept_analysis: ConceptAnalysis) -> List[ScenePlan]:
        """Generate scene plans from concept analysis"""

        user_message = f"Analyze this STEM concept and create scene plans:\n\n{json.dumps(concept_analysis.model_dump(), indent=2)}"

        try:
            response_json = self._call_llm_structured(
                system_prompt=self.SCENE_PLANNING_PROMPT,
                user_message=user_message,
                temperature=self.config.temperature,
                max_retries=3
            )

            # Parse and validate scene plans
            scene_plans = []
            for plan_data in response_json.get("scene_plans", []):
                try:
                    scene_plan = ScenePlan(**plan_data)
                    scene_plans.append(scene_plan)
                except Exception as e:
                    self.logger.warning(f"Invalid scene plan data: {e}")
                    continue

            return scene_plans

        except Exception as e:
            self.logger.error(f"Scene planning failed: {e}")
            raise ValueError(f"Failed to generate scene plans: {e}")

    def _generate_scene_codes(self, scene_plans: List[ScenePlan]) -> List[ManimSceneCode]:
        """Generate Manim code for each scene plan in parallel"""

        scene_codes = []
        self.logger.info(f"Starting parallel code generation for {len(scene_plans)} scenes")

        def generate_single_scene_code(scene_plan: ScenePlan) -> Optional[ManimSceneCode]:
            """Generate code for a single scene"""
            try:
                self.logger.info(f"Generating code for scene: {scene_plan.title}")

                class_name = self._sanitize_class_name(scene_plan.id)
                user_message = f"Generate Manim code for this scene plan:\n\n{json.dumps(scene_plan.model_dump(), indent=2)}"

                response = self._call_llm(
                    system_prompt=self.CODE_GENERATION_PROMPT.format(
                        scene_plan=json.dumps(scene_plan.model_dump(), indent=2),
                        class_name=class_name,
                        target_duration="15-30"
                    ),
                    user_message=user_message,
                    temperature=self.config.temperature,
                    max_retries=3
                )

                manim_code, extraction_method = self._extract_manim_code(response)

                if manim_code:
                    self._save_scene_code(scene_plan.id, class_name, manim_code, response)

                    scene_code = ManimSceneCode(
                        scene_id=scene_plan.id,
                        scene_name=class_name,
                        manim_code=manim_code,
                        raw_llm_output=response,
                        extraction_method=extraction_method
                    )

                    self.logger.info(f"Successfully generated code for scene: {class_name}")
                    return scene_code
                else:
                    self.logger.error(f"Failed to extract Manim code from response for scene: {scene_plan.id}")
                    return None

            except Exception as e:
                self.logger.error(f"Code generation failed for scene {scene_plan.id}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=min(len(scene_plans), 10)) as executor:
            future_to_plan = {executor.submit(generate_single_scene_code, plan): plan for plan in scene_plans}
            
            for future in as_completed(future_to_plan):
                scene_plan = future_to_plan[future]
                try:
                    result = future.result()
                    if result:
                        scene_codes.append(result)
                except Exception as e:
                    self.logger.error(f"Exception in parallel code generation for {scene_plan.id}: {e}")

        scene_codes.sort(key=lambda x: scene_plans.index(next(p for p in scene_plans if p.id == x.scene_id)))
        self.logger.info(f"Parallel code generation complete: {len(scene_codes)}/{len(scene_plans)} succeeded")

        return scene_codes

    def _extract_manim_code(self, response: str) -> tuple[str, str]:
        """Extract Manim code from LLM response using <manim> tags"""

        # Method 1: Try to extract from <manim>...</manim> tags
        manim_pattern = r'<manim>(.*?)</manim>'
        matches = re.findall(manim_pattern, response, re.DOTALL)

        if matches:
            # Take the first (most complete) match
            manim_code = matches[0].strip()
            # Clean the code by removing backticks
            manim_code = self._clean_manim_code(manim_code)
            return manim_code, "tags"

        # Method 2: Try to extract class definition if no tags found
        class_pattern = r'class\s+(\w+)\s*\(\s*Scene\s*\):.*?(?=\n\nclass|\Z)'
        matches = re.findall(class_pattern, response, re.DOTALL)

        if matches:
            # Find the complete code block
            class_start = response.find(f"class {matches[0]}(")
            if class_start != -1:
                # Find the end of this class (next class or end of response)
                remaining = response[class_start:]
                next_class = re.search(r'\n\nclass\s+\w+', remaining)
                if next_class:
                    manim_code = remaining[:next_class.start()]
                else:
                    manim_code = remaining

                # Add imports if missing
                if "from manim import" not in manim_code:
                    manim_code = "from manim import *\n\n" + manim_code

                # Clean the code by removing backticks
                manim_code = self._clean_manim_code(manim_code)
                return manim_code.strip(), "parsing"

        # Method 3: Last resort - try to fix common formatting issues
        if "class" in response and "def construct" in response:
            # Basic cleanup
            cleaned = response.strip()
            if not cleaned.startswith("from"):
                cleaned = "from manim import *\n\n" + cleaned

            # Clean the code by removing backticks
            cleaned = self._clean_manim_code(cleaned)
            return cleaned, "cleanup"

        return "", "failed"

    def _clean_manim_code(self, code: str) -> str:
        """Clean Manim code by removing backticks and fixing common issues"""

        # Remove all backticks - this is the main issue
        code = code.replace('`', '')

        # Fix common triple-backtick code block markers that might leave extra formatting
        code = re.sub(r'python\n', '', code, flags=re.IGNORECASE)
        code = re.sub(r'\npython', '', code, flags=re.IGNORECASE)

        # Remove any remaining markdown-style code formatting
        code = re.sub(r'^```.*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```.*$', '', code, flags=re.MULTILINE)

        # Clean up any double newlines that might have been created
        code = re.sub(r'\n{3,}', '\n\n', code)

        # Strip leading/trailing whitespace
        code = code.strip()

        # Log the cleaning if significant changes were made
        original_length = len(code.replace('`', ''))
        if original_length != len(code):
            self.logger.debug("Applied Manim code cleaning (removed backticks and formatting)")

        return code

    def _sanitize_class_name(self, scene_id: str) -> str:
        """Convert scene ID to valid Python class name"""
        # Remove invalid characters and convert to PascalCase
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', scene_id)
        # Capitalize first letter and ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = "Scene_" + sanitized
        sanitized = sanitized.title().replace('_', '')

        # Ensure it's not empty
        if not sanitized:
            sanitized = "AnimationScene"

        return sanitized

    def _save_scene_code(self, scene_id: str, class_name: str, manim_code: str, raw_output: str) -> Path:
        """Save generated Manim code to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene_id}_{class_name}_{timestamp}.py"
        filepath = self.output_dir / "scene_codes" / filename

        # Save both the clean code and raw output for debugging
        with open(filepath, 'w') as f:
            f.write(f"# Generated Manim code for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(manim_code)

        # Also save raw LLM output
        raw_filepath = filepath.with_suffix('.raw.txt')
        with open(raw_filepath, 'w') as f:
            f.write(f"# Raw LLM output for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(raw_output)

        return filepath

    def _render_scenes(self, scene_codes: List[ManimSceneCode]) -> List[RenderResult]:
        """Render each scene using ManimRenderer"""

        render_results = []

        for scene_code in scene_codes:
            self.logger.info(f"Rendering scene: {scene_code.scene_name}")

            # Generate output filename
            output_filename = f"{scene_code.scene_id}_{scene_code.scene_name}.mp4"

            try:
                # Use renderer to create the video
                render_result = self.renderer.render(
                    manim_code=scene_code.manim_code,
                    scene_name=scene_code.scene_name,
                    output_filename=output_filename
                )

                # Convert to our RenderResult format
                result = RenderResult(
                    scene_id=scene_code.scene_id,
                    success=render_result.success,
                    video_path=render_result.video_path,
                    error_message=render_result.error_message,
                    duration=render_result.duration,
                    resolution=render_result.resolution,
                    render_time=render_result.render_time
                )

                render_results.append(result)

                if result.success:
                    self.logger.info(f"Successfully rendered: {scene_code.scene_name}")
                    self.logger.info(f"  Video path: {result.video_path}")
                    self.logger.info(f"  Duration: {result.duration}s")
                else:
                    self.logger.error(f"Failed to render {scene_code.scene_name}: {result.error_message}")

            except Exception as e:
                self.logger.error(f"Rendering failed for {scene_code.scene_name}: {e}")
                render_results.append(RenderResult(
                    scene_id=scene_code.scene_id,
                    success=False,
                    error_message=str(e)
                ))

        return render_results

    def _concatenate_scenes(self, render_results: List[RenderResult]) -> Optional[Path]:
        """Concatenate rendered scenes into single silent animation"""

        if not render_results:
            self.logger.error("No render results to concatenate")
            return None

        # Get video paths and convert to absolute paths
        video_paths = []
        for r in render_results:
            if r.success and r.video_path:
                video_path = Path(r.video_path)
                if not video_path.is_absolute():
                    video_path = (Path.cwd() / video_path).resolve()
                if video_path.exists():
                    video_paths.append(video_path)
                else:
                    self.logger.warning(f"Video path does not exist: {video_path}")

        if not video_paths:
            self.logger.error("No successfully rendered scenes with valid video paths to concatenate")
            self.logger.error(f"Render results: {[(r.scene_id, r.success, r.video_path) for r in render_results]}")
            return None

        self.logger.info(f"Found {len(video_paths)} videos to concatenate")

        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"animation_{timestamp}.mp4"
            output_path = self.output_dir / "animations" / output_filename

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use FFmpeg to concatenate videos
            self.logger.info(f"Concatenating {len(video_paths)} scenes into {output_filename}")

            # Create a temporary file with list of input videos (use absolute paths)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                for video_path in video_paths:
                    # Ensure absolute path and escape single quotes
                    abs_path = str(video_path.resolve())
                    temp_file.write(f"file '{abs_path}'\n")
                    self.logger.debug(f"Adding to concat list: {abs_path}")
                temp_file_path = temp_file.name

            try:
                # FFmpeg concat command with absolute paths
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(temp_file_path),
                    "-c", "copy",
                    "-y",  # Overwrite output file if exists
                    str(output_path.resolve())
                ]

                self.logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0 and output_path.exists():
                    self.logger.info(f"Successfully concatenated animation: {output_filename}")
                    self.logger.info(f"Final video path: {output_path}")
                    return output_path
                else:
                    self.logger.error(f"FFmpeg concatenation failed with return code {result.returncode}")
                    self.logger.error(f"STDERR: {result.stderr}")
                    self.logger.error(f"STDOUT: {result.stdout}")
                    self.logger.error(f"Output path exists: {output_path.exists()}")
                    return None

            finally:
                # Clean up temporary file
                try:
                    import os
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        except Exception as e:
            self.logger.error(f"Scene concatenation failed: {e}")
            return None

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about animation generation performance"""
        return {
            "token_usage": self.get_token_usage(),
            "model_used": self.model,
            "reasoning_tokens": self.reasoning_tokens,
            "config": self.config.model_dump(),
            "renderer_status": "ready" if self.renderer.validate_manim_installation() else "not_ready"
        }