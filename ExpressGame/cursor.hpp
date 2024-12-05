#ifndef CURSOR_H
#define CURSOR_H

#include "raylib.h"

typedef struct Cursor {
    Texture2D texture;
    Rectangle rect;
    Vector2 position;
} Cursor;

#endif