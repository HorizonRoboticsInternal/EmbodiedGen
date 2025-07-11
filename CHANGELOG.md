# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0).


## [0.1.1] - 2025-07-11
### Feature
- Added intelligent quality checker and auto-retry pipeline for `image-to-3d` and `text-to-3d`.
- Added unit tests for quality checkers.
- `text-to-3d` now supports more text-to-image models, pipeline success rate improved to 94%.

## [0.1.0] - 2025-07-04
### Feature
🖼️ Single Image to Realistic 3D Asset
- Generates watertight, simulation-ready 3D meshes with physical attributes.
- Auto-labels semantic and quality tags (geometry, texture, foreground, etc.).
- Produces 2K textures with highlight removal and multi-view fusion.

📝 Text-to-3D Asset Generation
- Creates realistic 3D assets from natural language (English & Chinese).
- Filters assets via QA tags to ensure visual and geometric quality.

🎨 Texture Generation & Editing
- Generates 2K textures from mesh and text with semantic alignment.
- Plug-and-play modules adapt text-to-image models for 3D textures.
- Supports realistic and stylized texture outputs, including text textures.

