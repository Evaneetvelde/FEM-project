SetFactory("OpenCASCADE");

// Paramètres
L = 10;  // Longueur
W = 10;  // Largeur
H = 3;   // Hauteur par étage
e = 0.2; // Épaisseur murs/dalles

// --- ÉTAGE 1 ---
Box(1) = {0, 0, 0, L, W, e};       // Sol 1 (Bois)
Box(2) = {0, 0, e, L, e, H-e};     // Mur Nord
Box(3) = {0, W-e, e, L, e, H-e};   // Mur Sud
Box(4) = {0, e, e, e, W-2*e, H-e}; // Mur Ouest
Box(5) = {L-e, e, e, e, W-2*e, H-e}; // Mur Est
Box(6) = {L/2, e, e, e, W/2, H-e};  // Mur intérieur 1

// --- ÉTAGE 2 ---
Box(7) = {0, 0, H, L, W, e};       // Sol 2 (Bois)
Box(8) = {0, 0, H+e, L, e, H-e};
Box(9) = {0, W-e, H+e, L, e, H-e};
Box(10) = {0, e, H+e, e, W-2*e, H-e};
Box(11) = {L-e, e, H+e, e, W-2*e, H-e};
Box(12) = {e, W/2, H+e, L/2, e, H-e}; // Mur intérieur 2 (différent)

// --- ÉTAGE 3 ---
Box(13) = {0, 0, 2*H, L, W, e};    // Sol 3 (Bois)
Box(14) = {0, 0, 2*H+e, L, e, H-e};
Box(15) = {0, W-e, 2*H+e, L, e, H-e};
Box(16) = {0, e, 2*H+e, e, W-2*e, H-e};
Box(17) = {L-e, e, 2*H+e, e, W-2*e, H-e};
Box(18) = {L/3, e, 2*H+e, e, W-2*e, H-e}; // Mur intérieur 3 (différent)

// Groupes Physiques
Physical Volume("Bois", 1) = {1, 7, 13};
Physical Volume("Murs", 2) = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18};

// Maillage (taille des éléments)
Mesh.MeshSizeMin = 0.5;
Mesh.MeshSizeMax = 0.8;
