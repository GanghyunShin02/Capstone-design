// Gmsh project created on Mon Aug 17 14:19:41 2026
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 2.2, 0.41, 0};
//+
Disk(2) = {0.2, 0.2, 0, 0.05, 0.05};
//+
Physical Curve("inlet", 1) = {4};
//+
Physical Curve("up", 2) = {3};
//+
Physical Curve("outlet", 3) = {2};
//+
Physical Curve("down", 4) = {1};
//+
Physical Curve("obstacle", 5) = {5};
//+
Physical Surface("fluid", 6) = {1};
//+
Physical Surface("obstacle_surf", 7) = {2};
//+
Field[1] = BoundaryLayer;
//+
Delete Field [1];
//+
Field[1] = Threshold;
//+
Field[2] = Ball;
//+
Field[2].Radius = 0.05;
//+
Field[2].VIn = 0.02;
//+
Field[2].XCenter = 0.2;
//+
Field[2].YCenter = 0.2;
//+
Field[1].DistMax = 0.15;
//+
Field[1].DistMin = 0.01;
//+
Field[1].InField = 2;
//+
Field[1].SizeMin = 0.01;
//+
Background Field = 1;
//+
Field[1].SizeMin = 0.0001;
//+
Field[1].DistMin = 0.1;
//+
Recursive Delete {
  Surface{2}; 
}
//+
Circle(5) = {0.2, 0.2, 0, 0.05, 0, 2*Pi};
//+
Recursive Delete {
  Curve{5}; 
}
//+
Disk(2) = {0.2, 0.2, 0, 0.05, 0.05};
//+
BooleanDifference{ Surface{1}; }{ Surface{2}; Delete; }
//+
Physical Curve("obstacle", 5) = {5};
//+
Field[2].Radius = 0.1;
//+
Field[2].VIn = 0.002;
//+
Field[1].SizeMin = 0.001;
//+
Delete Field [2];
//+
Field[2] = Ball;
//+
Field[2].Radius = 0.1;
//+
Field[2].XCenter = 0.2;
//+
Field[2].YCenter = 0.2;
//+
Field[1].SizeMax = 0.1;
//+
Recursive Delete {
  Surface{1}; 
}
//+
Field[1].DistMax = 0.3;
//+
Field[1].DistMax = 1;
//+
Field[1].DistMin = 0.5;
//+
Physical Surface("surface", 6) = {2};
//+
Physical Surface("fluid", 8) = {2};
//+
Field[2].Radius = 0.05;
//+
Delete Field [2];
//+
Delete Field [1];
//+
Field[1] = Ball;
//+
Field[1].Radius = 0.05;
//+
Field[1].XCenter = 0.2;
//+
Field[1].YCenter = 0.2;
//+
Background Field = -1;
//+
Field[2] = Threshold;
//+
Field[2].DistMax = 1.5;
//+
Field[2].InField = 1;
//+
Field[2].SizeMax = 0.1;
//+
Field[2].SizeMin = 0.001;
//+
Background Field = 2;
