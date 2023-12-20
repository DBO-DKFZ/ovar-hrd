//// Simple Tissue Detection with Qupath 0.3.2

// Defaults
setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581", "Background" : " 255 255 255"}');

// Files
String annotationFolder = "path/to/export/folder"
String slidePath = getCurrentImageData().getServer().getPath()
String slideName = slidePath.split("/")[-1].split('\\.')[0]

runPlugin('qupath.imagej.detect.tissue.SimpleTissueDetection2', '{"threshold": 220,  "requestedPixelSizeMicrons": 200.0,  "minAreaMicrons": 1000000.0,  "maxHoleAreaMicrons": 1000000.0,  "darkBackground": false,  "smoothImage": false,  "medianCleanup": false,  "dilateBoundaries": false,  "smoothCoordinates": true,  "excludeOnBoundary": true,  "singleAnnotation": true}');


if (getAnnotationObjects().size == 0) {
    runPlugin('qupath.imagej.detect.tissue.SimpleTissueDetection2', '{"threshold": 220,  "requestedPixelSizeMicrons": 200.0,  "minAreaMicrons": 1000000.0,  "maxHoleAreaMicrons": 1000000.0,  "darkBackground": false,  "smoothImage": false,  "medianCleanup": false,  "dilateBoundaries": false,  "smoothCoordinates": true,  "excludeOnBoundary": false,  "singleAnnotation": true}');
}

if (getAnnotationObjects().size == 0) {
    print("No tissue: $slideName")
    return
}


def annotations = getAnnotationObjects()
def path = buildFilePath(annotationFolder, "${slideName}.geojson")

// Save
// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
exportObjectsToGeoJson(annotations, path, "FEATURE_COLLECTION")

print("Done: $slideName")