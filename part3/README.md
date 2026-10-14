# Final presentation

[Download the editable presentation](../output/presentations/Case_Study_Retention_and_Revenue_Growth_2026-10-14.pptx).

31 slides: 23 narrative slides, a closing discussion slide, and 7 appendix slides. Speaker notes accompany every slide. Slides 10–12 list all 45 selected features. Slide 16 explains score × annual contract value. Slides 18–19 report all three synthetic populations.

The checked-in PPTX is ready to review. The Python analysis does not require presentation tooling. To rebuild, use Node.js with OpenAI Artifact Tool and the Codex presentation skill installed. Set `ARTIFACT_TOOL_ENTRY` to the module path (or install `@oai/artifact-tool`), `PRESENTATION_SKILL_DIR` to the installed skill directory, and `RUNTIME_PYTHON` to Python.

```sh
python part3/build_model_demo_bundle.py
node part3/build_model_demo_presentation.mjs
STORY_FINAL_PATH=output/presentations/Case_Study_Rebuilt.pptx node part3/publish_model_demo_presentation.mjs
```

The finalizer writes a new file. The existing final deck remains unchanged. Export validation receipts are in `output/validation/`. They record the export-time state, rather than a later source-code revision.
