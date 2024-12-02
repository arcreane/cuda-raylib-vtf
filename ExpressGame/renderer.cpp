#include "renderer.hpp"

// Constructor: initializes Raylib and sets up the window
Renderer::Renderer(int width, int height, const char* title)
    : screen_width(width), screen_height(height) {
    InitWindow(screen_width, screen_height, title);
    SetTargetFPS(60); // Set desired frame rate
}

// Destructor: closes the Raylib window
Renderer::~Renderer() {
    CloseWindow();
}

// Draws particles on the screen
void Renderer::draw_particles(const std::vector<Particle>& particles) {
    BeginDrawing();
    ClearBackground(BLACK); // Set background color

    for (const auto& particle : particles) {
        DrawCircle(static_cast<int>(particle.x),
            static_cast<int>(particle.y),
            3.0f, // Particle radius
            particle.color);
    }

    EndDrawing();
}

// Displays the updated frame
void Renderer::display() {
    // Additional UI or overlays can be added here if needed
    // This method can be expanded for gamification features
}

// Get the screen width
int Renderer::get_screen_width() const {
    return screen_width;
}

// Get the screen height
int Renderer::get_screen_height() const {
    return screen_height;
}
