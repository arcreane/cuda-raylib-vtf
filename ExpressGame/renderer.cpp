#include "renderer.hpp"

// Fonction pour initialiser la fenêtre et les ressources
void InitGameWindow(int screenWidth, int screenHeight) {
    InitWindow(screenWidth, screenHeight, "A game about particles");
}

// Fonction pour afficher l'écran de victoire
void DrawVictoryScreen(int screenWidth, int screenHeight) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("VICTOIRE !", screenWidth / 2 - 100, screenHeight / 2 - 20, 40, GREEN);
    DrawText("Appuyez sur Echap pour quitter.", screenWidth / 2 - 150, screenHeight / 2 + 30, 20, WHITE);
    EndDrawing();
}

