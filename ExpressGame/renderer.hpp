#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <vector>
#include <raylib.h>
#include "particle.hpp"

class Renderer {
public:
    // Constructor and Destructor
    Renderer(int width, int height, const char* title);
    ~Renderer();

    // Rendering methods
    void draw_particles(const std::vector<Particle>& particles);
    void display();

    // Get screen dimensions
    int get_screen_width() const;
    int get_screen_height() const;

private:
    int screen_width;
    int screen_height;
};

#endif // RENDERER_HPP
