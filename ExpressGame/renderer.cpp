#include "raylib.h"
#include "renderer.hpp"

// Fonction pour initialiser la fenêtre et les ressources
void InitGameWindow(int screenWidth, int screenHeight) {
    InitWindow(screenWidth, screenHeight, "A game about particles");
}

// Fonction pour afficher l'écran de victoire
void DrawVictoryScreen(int screenWidth, int screenHeight) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("VICTOIRE !", screenWidth / 2 - MeasureText("VICTOIRE !", 40) / 2, screenHeight / 2 - 100, 40, GREEN);
    DrawText("Quitter [Echap]", screenWidth / 2 - MeasureText("Quitter [Echap]", 20) / 2, screenHeight / 2 + 40, 20, PURPLE);
    DrawText("Rejouer [R]", screenWidth / 2 - MeasureText("Rejouer [R]", 20) / 2, screenHeight / 2 + 70, 20, SKYBLUE);
    EndDrawing();
}

// Fonction pour afficher l'écran de défaite
void DrawDefeatScreen(int screenWidth, int screenHeight) {
	BeginDrawing();
	ClearBackground(BLACK);
	DrawText("DEFAITE !", screenWidth / 2 - MeasureText("DEFAITE !", 40)/2, screenHeight / 2 - 100, 40, RED);
    DrawText("Appuyez sur [Echap] pour quitter.", screenWidth / 2 - 150, screenHeight / 2 + 30, 20, PURPLE);
    DrawText("Appuyez sur [R] pour rejouer", screenWidth / 2 - 150, screenHeight / 2 + 60, 20, SKYBLUE);
	EndDrawing();
}

