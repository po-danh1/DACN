import logging
import time
import re
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from google import genai
from google.genai import types
    
from pydantic import BaseModel, Field


class SRTSubtitle(BaseModel):
    """Single subtitle entry in SRT format"""
    sequence: int
    start_time: str  # Format: HH:MM:SS,mmm
    end_time: str    # Format: HH:MM:SS,mmm
    text: str


class ScriptResult(BaseModel):
    """Result of script generation"""
    success: bool
    script_path: Optional[str] = None
    srt_content: Optional[str] = None
    subtitles: List[SRTSubtitle] = Field(default_factory=list)
    total_duration: Optional[float] = None
    error_message: Optional[str] = None

    # Metadata
    generation_time: Optional[float] = None
    video_duration: Optional[float] = None
    model_used: str = "gemini-2.5-flash"


class ScriptGenerator:
    """
    Script Generator: Analyzes silent animations and generates timestamped narration scripts
    using Gemini 2.5 Flash for multimodal video understanding.
    """

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.8,
        max_retries: int = 3,
        timeout: int = 180
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    SCRIPT_GENERATION_PROMPT = """You are an Educational Script Generator for STEM animations.

**TASK**: Watch the provided silent animation video and create a synchronized narration script in SRT (SubRip) format.

**VIDEO CONTEXT**: This is a mathematical/STEM educational animation explaining a concept step-by-step. The animation was generated to help students understand complex visual ideas.

**YOUR GUIDELINES**:

1. **Watch Carefully**: Analyze the entire video to understand:
   - What concepts are being visualized
   - The timing and flow of animations
   - When key elements appear or transform
   - The educational narrative being shown

2. **Create Educational Narration**:
   - Use clear, conversational language appropriate for students
   - Explain what's happening on screen as it happens
   - Define technical terms simply when they first appear
   - Connect visual elements to the underlying concepts
   - Use a friendly, engaging tone (not robotic)

3. **SRT FORMAT - EXTREMELY IMPORTANT**:
   - **TIMESTAMP FORMAT**: MUST be exactly HH:MM:SS,mmm (two digits for hours, minutes, seconds, COMMA, three digits milliseconds)
   - **EXAMPLE CORRECT**: 00:00:03,500 (COMMA between seconds and milliseconds!)
   - **WRONG EXAMPLES**: 00:00:03.500 (period), 00:00:03,50 (not 3 digits), 00:0:03,500 (missing leading zero)
   - **EACH timestamp MUST have**: HH:MM:SS,mmm format with COMMA as separator
   - **Each subtitle**: 3-6 seconds minimum for natural speech timing
   - **Complete sentences only**: Each subtitle must be a full, grammatically correct sentence
   - **Text per line**: Maximum 42 characters, maximum 2 lines per subtitle
   - **Sequence**: Start at 1 and increment by 1 for each subtitle

4. **Timing Guidelines**:
   - Start narration 0.5-1 second before visual elements appear
   - Allow 3-6 seconds per subtitle for comfortable speaking pace
   - Leave 0.5-1 second between subtitles for natural pauses
   - Continue explaining while animations progress
   - End explanations slightly before visual transitions

5. **Content Quality**:
   - Focus on explaining the "what" and "why" of visual elements
   - Connect visual steps to the overall concept
   - Use simple, clear language appropriate for the topic
   - Ensure smooth narrative flow between subtitles
   
**MORE TIMESTAMP EXAMPLES** (MUST follow this exact format with COMMA):
- 00:00:01,000 (1 second) - NOTE THE COMMA!
- 00:00:15,500 (15.5 seconds) - NOTE THE COMMA!
- 00:01:02,750 (1 minute, 2.75 seconds) - NOTE THE COMMA!
- 00:12:34,900 (12 minutes, 34.9 seconds) - NOTE THE COMMA!

**CRITICAL: ALWAYS USE COMMA (,) NOT PERIOD (.) BETWEEN SECONDS AND MILLISECONDS**
- ✅ CORRECT: 00:00:03,500 (COMMA!)
- ❌ WRONG: 00:00:03.500 (PERIOD!)

**COMMON MISTAKES TO AVOID**:
- ❌ 00:0:03,500 → ✅ 00:00:03,500 (missing leading zeros)
- ❌ 00:00:03,5 → ✅ 00:00:03,500 (not 3 decimal places)
- ❌ 0:00:03,500 → ✅ 00:00:03,500 (missing leading zero for hours)
- ❌ 00:00:03.500 → ✅ 00:00:03,500 (PERIOD instead of COMMA!)
- ❌ 00:00:03,500 → ✅ 00:00:03,500 (missing COMMA separator)
- ❌ 00:00:01,800 --> 00:00:02,100 (only 0.3 seconds - too short)
- ✅ 00:00:01,800 --> 00:00:05,200 (3.4 seconds - good for speech)

**REQUIREMENTS**:
- Output ONLY the SRT content wrapped in `<srt>` tags
- Every subtitle MUST be a complete sentence
- Use proper SRT timestamp format (HH:MM:SS,mmm) with COMMA separator and leading zeros
- Minimum 3 seconds per subtitle for natural speech
- Make sure that the timestamp and the generated narative MUST MATCH the visual element being shown, not too early, not too late.
- Make explanations educational and engaging
- Each segment should short, avoid long and unnecessary statements.
- Segments timeline should not be too close together, give them 1 to 2 seconds before starting a new one 

**SRT FORMAT EXAMPLES**:
<srt>
1
00:00:00,000 --> 00:00:04,500
Let's explore how bubble sort works step by step.

2
00:00:05,000 --> 00:00:08,500
We begin with an unsorted array of numbers.

3
00:00:09,000 --> 00:00:13,500
The algorithm compares adjacent elements in each pass.

4
00:00:14,000 --> 00:00:18,000
If elements are in wrong order, we swap their positions.

5
00:00:18,500 --> 00:00:23,000
This process continues until the array is fully sorted.
</srt>

Now analyze the video and generate the narration script following these exact requirements."""

    def execute(self, video_path: str) -> ScriptResult:
        """
        Generate narration script for a silent animation video

        Args:
            video_path: Path to the silent animation video file

        Returns:
            ScriptResult with generated SRT content and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting script generation for: {video_path}")

        try:
            # Validate video file
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Get video duration for context
            video_duration = self._get_video_duration(video_file)
            self.logger.info(f"Video duration: {video_duration:.2f} seconds")

            # Generate script using Gemini
            srt_content = self._generate_script_with_gemini(video_file)

            if srt_content:
                # Parse and validate SRT content
                subtitles = self._parse_srt_content(srt_content)
                script_duration = self._calculate_script_duration(subtitles)

                # Save script to file
                script_path = self._save_script(srt_content, video_file.stem)

                generation_time = time.time() - start_time
                self.logger.info(f"Script generation completed in {generation_time:.2f}s")
                self.logger.info(f"Generated {len(subtitles)} subtitles")
                self.logger.info(f"Script duration: {script_duration:.2f}s")

                return ScriptResult(
                    success=True,
                    script_path=str(script_path),
                    srt_content=srt_content,
                    subtitles=subtitles,
                    total_duration=script_duration,
                    generation_time=generation_time,
                    video_duration=video_duration,
                    model_used=self.model
                )
            else:
                raise ValueError("Failed to generate script content")

        except Exception as e:
            self.logger.error(f"Script generation failed: {e}")
            return ScriptResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used=self.model
            )

    def _generate_script_with_gemini(self, video_file: Path) -> Optional[str]:
        """Generate script using Gemini 2.5 Flash"""

        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Uploading video to Gemini (attempt {attempt + 1})")

                # Upload video file to Gemini
                uploaded_file = self.client.files.upload(file=str(video_file))

                # Wait for file to be processed
                self.logger.info("Waiting for file processing...")
                while uploaded_file.state == "PROCESSING":
                    time.sleep(2)
                    uploaded_file = self.client.files.get(name=uploaded_file.name)

                if uploaded_file.state == "FAILED":
                    raise ValueError(f"File processing failed: {uploaded_file.state}")

                if uploaded_file.state != "ACTIVE":
                    raise ValueError(f"Unexpected file state: {uploaded_file.state}")

                self.logger.info(f"File uploaded successfully: {uploaded_file.name}")

                # Generate script with Gemini
                self.logger.info(f"Generating script with {self.model}")
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded_file, self.SCRIPT_GENERATION_PROMPT],
                    config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=1024))
                )

                script_content = response.text
                self.logger.info("Received script content from Gemini")

                # Extract SRT content from <srt> tags
                srt_content = self._extract_srt_from_response(script_content)

                if srt_content:
                    return srt_content
                else:
                    self.logger.warning(f"No SRT content found in response (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        raise ValueError("Could not extract SRT content from Gemini response")

            except Exception as e:
                self.logger.warning(f"Gemini API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return None

    def _extract_srt_from_response(self, response: str) -> Optional[str]:
        """Extract SRT content from response wrapped in <srt> tags"""

        # Method 1: Extract from <srt>...</srt> tags
        srt_pattern = r'<srt>(.*?)</srt>'
        matches = re.findall(srt_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            srt_content = matches[0].strip()
            self.logger.info("Extracted SRT content from <srt> tags")
            return srt_content

        # Method 2: Look for SRT-like content without tags
        srt_lines = []
        lines = response.strip().split('\n')

        current_subtitle = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_subtitle:
                    srt_lines.extend(current_subtitle)
                    srt_lines.append("")  # Empty line between subtitles
                    current_subtitle = []
                continue

            # Check if line looks like SRT content
            if (line.isdigit() or  # Sequence number
                '-->' in line or    # Timestamp
                any(char in line for char in '.!?')):  # Text content
                current_subtitle.append(line)

        if current_subtitle:
            srt_lines.extend(current_subtitle)

        if srt_lines:
            srt_content = '\n'.join(srt_lines).strip()
            self.logger.info("Extracted SRT content without tags")
            return srt_content

        self.logger.warning("No SRT content found in response")
        return None

    def _parse_srt_content(self, srt_content: str) -> List[SRTSubtitle]:
        """Parse SRT content into structured subtitles"""

        subtitles = []
        lines = srt_content.strip().split('\n')

        i = 0
        while i < len(lines):
            try:
                # Skip empty lines
                if not lines[i].strip():
                    i += 1
                    continue

                # Parse sequence number (handle "12. timestamp" or just "12")
                sequence_line = lines[i].strip()
                
                # Check if sequence and timestamp are on same line (e.g., "12. 00:00:00,000 --> 00:00:05,000")
                if '-->' in sequence_line:
                    parts = sequence_line.split('.')
                    if len(parts) >= 2 and parts[0].strip().isdigit():
                        sequence = int(parts[0].strip())
                        timestamp_line = '.'.join(parts[1:]).strip()
                        
                        try:
                            start_time, end_time = [t.strip() for t in timestamp_line.split('-->')]
                        except ValueError:
                            self.logger.warning(f"Invalid timestamp format: {timestamp_line}")
                            i += 1
                            continue
                        
                        i += 1
                        
                        # Parse text lines
                        text_lines = []
                        while i < len(lines) and lines[i].strip() and not lines[i].strip().split('.')[0].isdigit() and '-->' not in lines[i]:
                            text_lines.append(lines[i].strip())
                            i += 1
                        
                        if text_lines:
                            text = ' '.join(text_lines)
                            subtitle = SRTSubtitle(
                                sequence=sequence,
                                start_time=start_time,
                                end_time=end_time,
                                text=text
                            )
                            subtitles.append(subtitle)
                        continue
                
                # Normal parsing: sequence on its own line
                if not sequence_line.isdigit():
                    self.logger.warning(f"Expected sequence number, got: {sequence_line}")
                    i += 1
                    continue

                sequence = int(sequence_line)
                i += 1

                # Parse timestamp line
                if i >= len(lines):
                    break

                timestamp_line = lines[i].strip()
                if '-->' not in timestamp_line:
                    self.logger.warning(f"Expected timestamp line, got: {timestamp_line}")
                    i += 1
                    continue

                try:
                    start_time, end_time = [t.strip() for t in timestamp_line.split('-->')]
                except ValueError:
                    self.logger.warning(f"Invalid timestamp format: {timestamp_line}")
                    i += 1
                    continue

                i += 1

                # Parse text lines (until empty line or next sequence)
                text_lines = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                    text_lines.append(lines[i].strip())
                    i += 1

                if text_lines:
                    text = ' '.join(text_lines)

                    subtitle = SRTSubtitle(
                        sequence=sequence,
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    )
                    subtitles.append(subtitle)

            except Exception as e:
                self.logger.warning(f"Error parsing subtitle at line {i}: {e}")
                i += 1

        self.logger.info(f"Parsed {len(subtitles)} subtitles from SRT content")
        return subtitles

    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize malformed timestamps to proper HH:MM:SS,mmm format"""
        
        timestamp = timestamp.strip()
        timestamp = timestamp.replace('.', ',').replace(':', ',')
        parts = timestamp.split(',')
        
        if len(parts) == 3:
            minutes, seconds, milliseconds = parts
            hours = "00"
        elif len(parts) == 4:
            hours, minutes, seconds, milliseconds = parts
        else:
            raise ValueError(f"Cannot parse timestamp: {timestamp}")
        
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        milliseconds = int(milliseconds)
        
        if minutes >= 60:
            hours += minutes // 60
            minutes = minutes % 60
        
        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60
        
        if milliseconds >= 1000:
            seconds += milliseconds // 1000
            milliseconds = milliseconds % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _calculate_script_duration(self, subtitles: List[SRTSubtitle]) -> float:
        """Calculate total duration of the script in seconds"""

        if not subtitles:
            return 0.0

        try:
            # Parse the end time of the last subtitle
            last_subtitle = max(subtitles, key=lambda s: s.sequence)
            end_time_str = self._normalize_timestamp(last_subtitle.end_time)

            # Convert HH:MM:SS,mmm to seconds
            time_parts = end_time_str.replace(',', ':').split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            milliseconds = int(time_parts[3]) if len(time_parts) > 3 else 0

            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            return total_seconds

        except Exception as e:
            self.logger.warning(f"Error calculating script duration: {e}")
            return 0.0

    def _save_script(self, srt_content: str, video_stem: str) -> Path:
        """Save SRT script to file with normalized timestamps"""

        normalized_content = self._normalize_srt_timestamps(srt_content)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_stem}_script_{timestamp}.srt"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(normalized_content)

        self.logger.info(f"Script saved to: {filepath}")
        return filepath

    def _normalize_srt_timestamps(self, srt_content: str) -> str:
        """Normalize all timestamps in SRT content"""
        
        lines = srt_content.split('\n')
        normalized_lines = []
        
        for line in lines:
            if '-->' in line:
                try:
                    parts = line.split('-->')
                    if len(parts) == 2:
                        start = self._normalize_timestamp(parts[0])
                        end = self._normalize_timestamp(parts[1])
                        normalized_lines.append(f"{start} --> {end}")
                    else:
                        normalized_lines.append(line)
                except Exception as e:
                    self.logger.warning(f"Could not normalize timestamp line: {line} - {e}")
                    normalized_lines.append(line)
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)

    def _get_video_duration(self, video_file: Path) -> float:
        """Get video duration using ffprobe or return default"""

        try:
            import subprocess

            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "default=nokey=1:noprint_wrappers=1",
                "-show_entries", "format=duration",
                str(video_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                duration_str = result.stdout.strip()
                return float(duration_str)
            else:
                self.logger.warning(f"ffprobe failed: {result.stderr}")
                return 120.0  # Default duration

        except Exception as e:
            self.logger.warning(f"Could not get video duration: {e}")
            return 120.0  # Default duration

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about script generation performance"""
        return {
            "model_used": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "output_dir": str(self.output_dir)
        }