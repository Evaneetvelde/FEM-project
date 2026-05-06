// Poelon a raclette creux dans un bloc d'air 20 x 20 x 20.
// Materiaux: air = 5, poelon = 11 (PTFE).

SetFactory("OpenCASCADE");

lc = 0.75;
epaisseur = 0.04;

Mesh.MeshSizeMin = lc;
Mesh.MeshSizeMax = lc;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

// Surface interieure basse du poelon.
Point(1) = {4, 4, 0, lc};
Point(2) = {-4, 4, 0, lc};
Point(3) = {-4, -4, 0, lc};
Point(4) = {4, -4, 0, lc};
Point(9) = {5, 5, 1.5, lc};
Point(15) = {-5, 5, 1.5, lc};
Point(11) = {-5, -5, 1.5, lc};
Point(12) = {5, -5, 1.5, lc};

// Surface exterieure, obtenue par translation verticale de l'interieur.
Point(5) = {4, 4, epaisseur, lc};
Point(6) = {-4, 4, epaisseur, lc};
Point(7) = {-4, -4, epaisseur, lc};
Point(8) = {4, -4, epaisseur, lc};
Point(19) = {5, 5, 1.5 + epaisseur, lc};
Point(16) = {-5, 5, 1.5 + epaisseur, lc};
Point(17) = {-5, -5, 1.5 + epaisseur, lc};
Point(18) = {5, -5, 1.5 + epaisseur, lc};

// Courbes interieures.
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {9, 15};
Line(6) = {15, 11};
Line(7) = {11, 12};
Line(8) = {12, 9};
Line(9) = {1, 9};
Line(10) = {2, 15};
Line(11) = {3, 11};
Line(12) = {4, 12};

// Courbes exterieures.
Line(13) = {5, 6};
Line(14) = {6, 7};
Line(15) = {7, 8};
Line(16) = {8, 5};
Line(17) = {19, 16};
Line(18) = {16, 17};
Line(19) = {17, 18};
Line(20) = {18, 19};
Line(21) = {5, 19};
Line(22) = {6, 16};
Line(23) = {7, 17};
Line(24) = {8, 18};

// Courbes du rebord ouvert.
Line(25) = {9, 19};
Line(26) = {15, 16};
Line(27) = {11, 17};
Line(28) = {12, 18};

// Faces interieures.
cl_i0 = newreg; Curve Loop(cl_i0) = {1, 2, 3, 4};
s_i0 = newreg; Plane Surface(s_i0) = {cl_i0};
cl_i1 = newreg; Curve Loop(cl_i1) = {1, 10, -5, -9};
s_i1 = newreg; Plane Surface(s_i1) = {cl_i1};
cl_i2 = newreg; Curve Loop(cl_i2) = {2, 11, -6, -10};
s_i2 = newreg; Plane Surface(s_i2) = {cl_i2};
cl_i3 = newreg; Curve Loop(cl_i3) = {3, 12, -7, -11};
s_i3 = newreg; Plane Surface(s_i3) = {cl_i3};
cl_i4 = newreg; Curve Loop(cl_i4) = {4, 9, -8, -12};
s_i4 = newreg; Plane Surface(s_i4) = {cl_i4};

// Faces exterieures.
cl_o0 = newreg; Curve Loop(cl_o0) = {13, 14, 15, 16};
s_o0 = newreg; Plane Surface(s_o0) = {cl_o0};
cl_o1 = newreg; Curve Loop(cl_o1) = {13, 22, -17, -21};
s_o1 = newreg; Plane Surface(s_o1) = {cl_o1};
cl_o2 = newreg; Curve Loop(cl_o2) = {14, 23, -18, -22};
s_o2 = newreg; Plane Surface(s_o2) = {cl_o2};
cl_o3 = newreg; Curve Loop(cl_o3) = {15, 24, -19, -23};
s_o3 = newreg; Plane Surface(s_o3) = {cl_o3};
cl_o4 = newreg; Curve Loop(cl_o4) = {16, 21, -20, -24};
s_o4 = newreg; Plane Surface(s_o4) = {cl_o4};

// Rebord superieur qui ferme la matiere, pas le volume d'air interieur.
cl_r1 = newreg; Curve Loop(cl_r1) = {5, 26, -17, -25};
s_r1 = newreg; Plane Surface(s_r1) = {cl_r1};
cl_r2 = newreg; Curve Loop(cl_r2) = {6, 27, -18, -26};
s_r2 = newreg; Plane Surface(s_r2) = {cl_r2};
cl_r3 = newreg; Curve Loop(cl_r3) = {7, 28, -19, -27};
s_r3 = newreg; Plane Surface(s_r3) = {cl_r3};
cl_r4 = newreg; Curve Loop(cl_r4) = {8, 25, -20, -28};
s_r4 = newreg; Plane Surface(s_r4) = {cl_r4};

sl_poelon = newreg;
Surface Loop(sl_poelon) = {s_i0, s_i1, s_i2, s_i3, s_i4, s_o0, s_o1, s_o2, s_o3, s_o4, s_r1, s_r2, s_r3, s_r4};
v_poelon = newreg;
Volume(v_poelon) = {sl_poelon};

// Poignee pleine.
Box(2000) = {5, -1.5, 1.5, 2.5, 3.0, epaisseur};

poelon_volumes[] = {v_poelon, 2000};

// Bloc d'air de cote 20 qui englobe tout le poelon.
Box(1000) = {-10, -10, -10, 20, 20, 20};
fragments[] = BooleanFragments{ Volume{1000}; Delete; }{ Volume{poelon_volumes[]}; Delete; };

Physical Volume("air", 5) = {2001};
Physical Volume("poelon_ptfe", 11) = {58, 2000};
