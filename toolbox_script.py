from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterDistance, QgsProcessingParameterFeatureSource, 
                       QgsProcessingParameterNumber, QgsProcessingParameterBoolean, QgsProcessingOutputString, QgsGeometry,
                       QgsFeature, QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsField, QgsProject, QgsWkbTypes,
                       QgsVectorLayer, QgsPointXY, QgsSpatialIndex, QgsRectangle, QgsUnitTypes, QgsFeatureRequest,
                       QgsVectorLayerUtils, QgsDistanceArea, QgsProcessingFeatureSourceDefinition, QgsProcessingException,
                       DEFAULT_SEGMENT_EPSILON, NULL, QgsProcessingParameterDefinition, QgsProcessingParameterField,
                       QgsVectorFileWriter, QgsFeatureSink, QgsProcessingUtils, QgsProperty, QgsProcessingParameterFeatureSink)
from qgis import processing
from difflib import get_close_matches
from re import search
from collections import Counter
from math import inf
from datetime import (date, datetime)
import networkx as nx
import time

class QualityControlAlgorithm(QgsProcessingAlgorithm):
    # This is an example algorithm that takes a vector layer, creates some new layers and returns some results.
    
    def tr(self, string):
        # Returns a translatable string with the self.tr() function.
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return QualityControlAlgorithm()

    def name(self):
        # Returns the unique algorithm name.
        return 'qualitycontrol'

    def displayName(self):
        # Returns the translated algorithm name.
        return self.tr('Quality Control for Waterway Networks')

    def group(self):
        # Returns the name of the group this algorithm belongs to.
        return self.tr('Quality Control')

    def groupId(self):
        # Returns the unique ID of the group this algorithm belongs to.
        return 'qaqc'

    def shortHelpString(self):
        #Returns a localised short help string for the algorithm.
        desc = '''The "Quality Control for Waterway Networks" Processing Algorithm is an automated process to update the U.S. Army Corps of Engineer’s (USACE) Engineer Research and Development Center (ERDC) Waterway Network. After a user introduces desired changes to a line layer representing the waterways, the algorithm creates a fully connected network, and controls topology quality.  In addition, the algorithm updates waterway depths and geometries based on the most recent version of the USACE National Channel Framework (NCF), and performs spatial joins of network nodes with other various sources of data.
        
        <hr>
        <b>INPUTS</b>
        
        '''
        desc += '• The <b>Input Waterway layer</b> is your input. Accepts line vector layers. Usually will be your Waterway Network layer.<br>'
        desc += '• The <b>ChannelReach layer</b> is the layer containing NCF data. Accepts polygon vector layers. Usually is named "ChannelReach".<br>'
        desc += '• The <b>State layer</b> is an optional layer containing data of various states. Accepts polygon vector layers. State abbreviations/FIPS codes will not be added to the node layer if left blank.<br>'
        desc += '• The <b>County layer</b> is an optional layer containing data of various counties. Accepts polygon vector layers. County names will not be added to the node layer if left blank.<br>'
        desc += '• The <b>Country layer</b> is an optional layer containing data of various countries. Accepts polygon vector layers. Country abbreviations will not be added to the node layer if left blank.<br>'
        desc += '''• If the <b>Bypass the "Split Waterway Lines Based On ChannelReach" step?</b> is checked, the geometries of the input layer will not change to reflect NCF geometries. Waterway network depths will still be updated. By checking this box, the execution of the algorithm is faster.

        At the time of developing this tool, public versions of the above-mentioned inputs can be found in the following links:<br>'''
        desc += '• NCF ChannelReach: <a href="https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer/2">https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer/2</a><br>'
        desc += '• U.S. States: <a href="https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip">https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip</a><br>'
        desc += '''• U.S. Counties: <a href="https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip">https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip</a>

        The user may substitute the State, County, and Country layers with other polygon layers. The algorithm associates a waterway network node with a polygon's area. Fields to be joined are selected within the Advanced Parameters.
        
        <hr>
        <b>SETTINGS</b>

        • The <b>Minimum geometry length</b> is the minimum desired length of a link, in miles (LenMiles). All links with LenMiles below this threshold will be deleted. Recommended value: 0.02
        '''
        desc += 'In the Advanced Parameters,<br>'
        desc += '• The <b>"Split Waterway Lines Based On ChannelReach" length</b> is the minimum length of a link separation, in miles. If a newly-created link with length below this threshold would be made during the "Split Waterway Lines Based On ChannelReach" step, it is instead undone. Recommended value: 0.03<br>'
        desc += '• The <b>Snapping tolerance</b> controls how close line endpoints need to be before they are snapped. If the Disconnected Islands step detects islands, this will be iteratively multiplied until no islands are detected. Recommended value: 0.0002<br>'
        desc += '• The <b>Disconnected Islands tolerance</b> detects any groups of links in the input layer that are not connected to each other ("islands") within the threshold. Recommended value: 0.000001<br>'
        desc += '''• All parameters in the <b>FIELDS</b> section are optional. Accepts any field types. Columns involving their respective layer will not be added to the node layer if left blank.
        
        The USACE-ERDC waterway network is non-authoritative product created by the Coastal and Hydraulics Laboratory (CHL) for research purposes.'''
        return self.tr(desc)

    def initAlgorithm(self, config=None):
        # Here we define the inputs and outputs of the algorithm.
        # https://gis.stackexchange.com/questions/377793/how-to-group-parameters-in-pyqgis-processing-plugin
        self.addParameter(QgsProcessingParameterFeatureSource('INPUT', self.tr('<br><b>INPUTS</b><br><br>Input Waterway layer'), types=[QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterFeatureSource('CHANNELREACH_INPUT', self.tr('ChannelReach layer'), types=[QgsProcessing.TypeVectorPolygon]))
        
        self.addParameter(QgsProcessingParameterFeatureSource('STATE_INPUT', self.tr('State layer'), types=[QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterFeatureSource('COUNTY_INPUT', self.tr('County layer'), types=[QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterFeatureSource('COUNTRY_INPUT', self.tr('Country layer'), types=[QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterBoolean('BYPASS_SPLIT', self.tr('Bypass the "Split Waterway Lines Based On ChannelReach" step?')))

        self.addParameter(QgsProcessingParameterNumber('MIN_LENGTH', self.tr('<hr><br><b>SETTINGS</b><br><br>Minimum geometry length (mi)'), type=1, defaultValue = 0.02, minValue=0.0))
        break_length = QgsProcessingParameterNumber('BREAK_LENGTH', self.tr('"Split Waterway Lines Based On ChannelReach" length (mi)'), type=1, defaultValue = 0.03, minValue=0.0)
        break_length.setFlags(break_length.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(break_length)
        snap_tol = QgsProcessingParameterDistance('SNAP_TOL', self.tr('Snapping tolerance'), defaultValue = 0.0002, parentParameterName='INPUT', minValue=0.0)
        snap_tol.setFlags(snap_tol.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(snap_tol)
        island_tol = QgsProcessingParameterDistance('ISLAND_TOL', self.tr('Disconnected Islands tolerance'), defaultValue = 0.000001, parentParameterName='INPUT', minValue=0.0)
        island_tol.setFlags(island_tol.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(island_tol)
        
        state_abbr = QgsProcessingParameterField('STATE_ABBR', '<hr><br><b>FIELDS</b><br><br>State Abbreviation', type=QgsProcessingParameterField.Any, parentLayerParameterName='STATE_INPUT', allowMultiple=False, optional=True)
        state_abbr.setFlags(state_abbr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(state_abbr)
        state_fips = QgsProcessingParameterField('STATE_FIPS', 'State FIPS', type=QgsProcessingParameterField.Any, parentLayerParameterName='STATE_INPUT', allowMultiple=False, optional=True)
        state_fips.setFlags(state_fips.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(state_fips)
        county_name = QgsProcessingParameterField('COUNTY_NAME', 'County Name', type=QgsProcessingParameterField.Any, parentLayerParameterName='COUNTY_INPUT', allowMultiple=False, optional=True)
        county_name.setFlags(county_name.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(county_name)
        county_fips = QgsProcessingParameterField('COUNTY_FIPS', 'County FIPS', type=QgsProcessingParameterField.Any, parentLayerParameterName='COUNTY_INPUT', allowMultiple=False, optional=True)
        county_fips.setFlags(county_fips.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(county_fips)
        country_abbr = QgsProcessingParameterField('COUNTRY_ABBR', 'Country Abbreviation', type=QgsProcessingParameterField.Any, parentLayerParameterName='COUNTRY_INPUT', allowMultiple=False, optional=True)
        country_abbr.setFlags(country_abbr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(country_abbr)

        self.addParameter(QgsProcessingParameterFeatureSink('OUTPUT', self.tr('<hr><br><b>OUTPUTS</b><br><br>Links output'), type=QgsProcessing.TypeVectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink('NODE_OUTPUT', self.tr('Nodes output'), type=QgsProcessing.TypeVectorPoint))
        self.addOutput(QgsProcessingOutputString('FLAGGED', self.tr('Flagged LinkIDs')))

    def processAlgorithm(self, parameters, context, feedback):
        # Here is where the processing itself takes place.            
        input_layer = self.parameterAsVectorLayer(parameters, 'INPUT', context)
        if input_layer is None or QgsWkbTypes.geometryType(input_layer.wkbType()) != QgsWkbTypes.LineGeometry or input_layer.featureCount() == 0:
            raise QgsProcessingException("Input layer failed to load!")
        
        state_exists = False
        if self.parameterAsVectorLayer(parameters, 'STATE_INPUT', context) != None:
            state_layer = self.parameterAsVectorLayer(parameters, 'STATE_INPUT', context)
            if state_layer is None or QgsWkbTypes.geometryType(state_layer.wkbType()) != QgsWkbTypes.PolygonGeometry or state_layer.featureCount() == 0:
                raise QgsProcessingException("State layer failed to load!")
            state_exists = True
            
        county_exists = False
        if self.parameterAsVectorLayer(parameters, 'COUNTY_INPUT', context) != None:
            county_layer = self.parameterAsVectorLayer(parameters, 'COUNTY_INPUT', context)
            if county_layer is None or QgsWkbTypes.geometryType(county_layer.wkbType()) != QgsWkbTypes.PolygonGeometry or county_layer.featureCount() == 0:
                raise QgsProcessingException("County layer failed to load!")
            county_exists = True
        
        country_exists = False
        if self.parameterAsVectorLayer(parameters, 'COUNTRY_INPUT', context) != None:
            country_layer = self.parameterAsVectorLayer(parameters, 'COUNTRY_INPUT', context)
            if country_layer is None or QgsWkbTypes.geometryType(country_layer.wkbType()) != QgsWkbTypes.PolygonGeometry or country_layer.featureCount() == 0:
                raise QgsProcessingException("Country layer failed to load!")
            country_exists = True
        
        bypass_split = self.parameterAsBoolean(parameters, 'BYPASS_SPLIT', context)
        min_geom_length = self.parameterAsDouble(parameters, 'MIN_LENGTH', context)
        break_length = self.parameterAsDouble(parameters, 'BREAK_LENGTH', context)
        snapping_tolerance = self.parameterAsDouble(parameters, 'SNAP_TOL', context)
        disconnected_islands_tolerance = self.parameterAsDouble(parameters, 'ISLAND_TOL', context)
        
        state_abbr = self.parameterAsString(parameters, 'STATE_ABBR', context)
        state_fips = self.parameterAsString(parameters, 'STATE_FIPS', context)
        county_name = self.parameterAsString(parameters, 'COUNTY_NAME', context)
        county_fips = self.parameterAsString(parameters, 'COUNTY_FIPS', context)
        country_abbr = self.parameterAsString(parameters, 'COUNTRY_ABBR', context) #next(iter(self.parameterAsStrings(parameters, 'COUNTRY_ABBR', context)), None)

        project = QgsProject.instance()
        if feedback.isCanceled():
            return {} 
        

        # Class DisconnectedIslands modified from disconnected-islands
        # Copyright (c) 2024 Peter Smythe (https://github.com/AfriGIS-South-Africa/disconnected-islands)
        # Licensed under MIT
        class DisconnectedIslands(object):
            def __init__(self, l):
                self.layer = l

            def run(self, tolerance):
                attrIdx = self.layer.fields().indexFromName("networkGrp")
                if attrIdx == -1:
                    self.layer.startEditing()
                    self.layer.dataProvider().addAttributes([QgsField("networkGrp", QVariant.Int)])
                    self.layer.commitChanges()
                    attrIdx = self.layer.fields().indexFromName("networkGrp")
                G = nx.Graph()
                if tolerance == 0:
                    tolerance = disconnected_islands_tolerance
                self.layer.startEditing()
                for feat in self.layer.getFeatures():    
                    self.layer.changeAttributeValue(feat.id(), attrIdx, -1)
                    geom = feat.geometry()
                    QgsGeometry.convertToSingleType(geom)
                    if not geom.isNull():
                        line = geom.asPolyline()
                        for i in range(len(line)-1):
                            G.add_edges_from([((int(line[i][0]/tolerance), int(line[i][1]/tolerance)), (int(line[i+1][0]/tolerance), int(line[i+1][1]/tolerance)), {'fid': feat.id()})]) 
                self.layer.commitChanges()
                connected_components = list(G.subgraph(c) for c in nx.connected_components(G))
                fid_comp = {}
                for i, graph in enumerate(connected_components):
                    for edge in graph.edges(data=True):
                        fid_comp[edge[2].get('fid', None)] = i
                countMap = {}
                for v in fid_comp.values():
                    countMap[v] = countMap.get(v,0) + 1
                isolated = [k for k, v in fid_comp.items() if countMap[v] == 1]
                self.layer.selectByIds(isolated)
                self.layer.startEditing()
                for (fid, group) in fid_comp.items():
                    self.layer.changeAttributeValue(fid, attrIdx, group)
                self.layer.commitChanges()
                return (self.layer, [i for i in set(fid_comp.values()) if i > 0])

        # Class Networks modified from networks
        # Copyright (c) [2025] crocovert (https://github.com/crocovert)
        # Licensed under GPL-3.0
        class Networks(QgsProcessingAlgorithm):

            def build_graph(self, parameters):       
                reseau = parameters["RESEAU"]
                sens = parameters["SENS"]
                ident = parameters["IDENT"]
                prefixe = parameters["PREFIXE"]
                dec = parameters["DECIMALES"]

                layer = reseau
                nom_champs = []
                for i in layer.fields():
                    nom_champs.append(i.name())
                if ("i" not in nom_champs):
                    layer.dataProvider().addAttributes([QgsField("i", QVariant.String)])
                if ("j" not in nom_champs):
                    layer.dataProvider().addAttributes([QgsField("j", QVariant.String)])
                if ("ij" not in nom_champs):
                    layer.dataProvider().addAttributes([QgsField("ij", QVariant.String)])
                layer.updateFields()

                ida = layer.fields().indexFromName("i")
                idb = layer.fields().indexFromName("j")
                idij = layer.fields().indexFromName("ij")
                noeuds = {}
                src = QgsCoordinateReferenceSystem(layer.crs())
                dest = QgsCoordinateReferenceSystem("EPSG:4326")
                xtr = QgsCoordinateTransform(src, dest, project)

                for ligne in layer.getFeatures():
                    if len(sens) == 0:
                        test_sens = '1'
                    else:
                        if ligne[sens] == '1':
                            test_sens = '1'
                        else:
                            test_sens = '0'
                    
                    gligne = ligne.geometry()
                    if test_sens == '1':
                        if gligne.wkbType() in [QgsWkbTypes.MultiLineString, QgsWkbTypes.MultiLineStringZ]:
                            g = gligne.asMultiPolyline()
                            na = g[0][0]
                            liba = str(int(xtr.transform(na)[0] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec) + str(int(xtr.transform(na)[1] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec)
                            if na.compare(g[-1][-1]):
                                nb = g[-1][-2]
                            else:
                                nb = g[-1][-1]
                            libb = str(int(xtr.transform(nb)[0] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec) + str(int(xtr.transform(nb)[1] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec)
                        elif gligne.wkbType() in [QgsWkbTypes.LineString, QgsWkbTypes.LineStringZ]:
                            g = gligne.asPolyline()
                            na = g[0]
                            liba = str(int(xtr.transform(na)[0] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec) + str(int(xtr.transform(na)[1] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec)
                            if na.compare(g[-1]):
                                nb = g[-2]
                            else:
                                nb = g[-1]
                            libb = str(int(xtr.transform(nb)[0] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec) + str(int(xtr.transform(nb)[1] * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))).zfill(dec)
                        else:
                            continue

                        if (na not in noeuds):
                            noeuds[na] = (prefixe + liba, 1)
                        else:
                            noeuds[na] = (prefixe + liba, noeuds[na][1] + 1)
                        if (nb not in noeuds):
                            noeuds[nb] = (prefixe + libb, 1)
                        else:
                            noeuds[nb] = (prefixe + libb, noeuds[nb][1] + 1)

                node_layer = QgsVectorLayer("Point", "temporary_points", "memory")
                node_provider = node_layer.dataProvider()
                node_provider.addAttributes([QgsField("num", QVariant.String), QgsField("nb", QVariant.Int)])
                node_layer.updateFields()

                node_layer.startEditing()
                for i, n in enumerate(noeuds):
                    node = QgsFeature()
                    node.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(n[0], n[1])))
                    if ident == 0:
                        noeuds[n] = (prefixe + str(i), noeuds[n][1])
                    node.setAttributes([noeuds[n][0], noeuds[n][1]])
                    node_provider.addFeatures([node])
                node_layer.commitChanges()
                
                layer.startEditing()
                for i, ligne in enumerate(layer.getFeatures()):
                    if len(sens) == 0:
                        test_sens = '1'
                    else:
                        if ligne[sens] == '1':
                            test_sens = '1'
                        else:
                            test_sens = '0'

                    if test_sens == '1':
                        gligne = ligne.geometry()
                        if gligne.wkbType() in [QgsWkbTypes.MultiLineString, QgsWkbTypes.MultiLineStringZ]:
                            g = gligne.asMultiPolyline()
                            na = g[0][0]
                            if na.compare(g[-1][-1]):
                                nb = g[-1][-2]
                            else:
                                nb = g[-1][-1]
                        elif gligne.wkbType() in [QgsWkbTypes.LineString, QgsWkbTypes.LineStringZ]:
                            g = gligne.asPolyline()
                            na = g[0]
                            if na.compare(g[-1]):
                                nb = g[-2]
                            else:
                                nb = g[-1]
                        else:
                            continue

                        if noeuds[na][0] != noeuds[nb][0]:
                            id = ligne.id()
                            valid = {ida : str(noeuds[na][0]), idb: str(noeuds[nb][0]), idij: str(noeuds[na][0] + "-" + noeuds[nb][0])}
                            layer.changeAttributeValues(id, valid)
                layer.commitChanges()
                return (layer, node_layer)
        
        def progressUpdate(percent, changelog = ""):
            feedback.setProgress(percent)
            if changelog != "":
                feedback.setProgressText(changelog)
        
        def CreateSpatialIndex(layer):
            spatialIndex = QgsSpatialIndex(layer.getFeatures())
            indexedFeatureIDs = spatialIndex.intersects(QgsRectangle(QgsPointXY(-inf, -inf), QgsPointXY(inf, inf)))
            temp_layer = QgsVectorLayer("MultiLineString?crs={}&index=yes".format(lines_layer.crs().authid()), "centroids", "memory")
            provider = temp_layer.dataProvider()
            provider.addAttributes(layer.fields())
            temp_layer.updateFields()
            temp_layer.startEditing()
            for id in indexedFeatureIDs:
                provider.addFeature(layer.getFeature(id))
            temp_layer.commitChanges()
            return temp_layer

        def makePointList(geom, list_of_points):
            if QgsWkbTypes.isSingleType(geom.wkbType()):
                    for pnt in geom.asPolyline():
                        if (pnt.y(), pnt.x()) not in list_of_points:
                            list_of_points.append((pnt.y(), pnt.x()))
            else:
                for part in geom.asMultiPolyline():
                    for pnt in part:
                        if (pnt.y(), pnt.x()) not in list_of_points:
                            list_of_points.append((pnt.y(), pnt.x()))

        def unionSplits(geom, feat_to_remove, split_layer, split_feats, new_feats):
            short_geom_spatial_index = QgsSpatialIndex(split_layer.getFeatures())
            neighborIDs = short_geom_spatial_index.nearestNeighbor(geom)
            small = [s.id() for s in split_layer.getFeatures() if distanceArea.convertLengthMeasurement(s.geometry().length(), QgsUnitTypes.DistanceMiles) <= break_length]
            new_feat_connectors = [(n, n.id()) for n in new_feats if n.geometry().intersects(geom)]

            if feat_to_remove.id() in neighborIDs:
                neighborIDs.remove(feat_to_remove.id())
            if feat_to_remove.id() in small:
                small.remove(feat_to_remove.id())
                split_layer.startEditing()
                split_layer.deleteFeature(feat_to_remove.id())
                split_layer.commitChanges()
            if feat_to_remove.id() in [x[1] for x in new_feat_connectors]:
                new_feat_cidx = [x[1] for x in new_feat_connectors].index(feat_to_remove.id())
                new_feat_connectors.pop(new_feat_cidx)

            actual_neighborIDs = [nID for nID in neighborIDs if split_layer.getFeature(nID).geometry().intersects(geom)]

            if actual_neighborIDs:
                n = split_layer.getFeature(actual_neighborIDs[0]).geometry()
                geom = geom.combine(n)
                split_layer.startEditing()
                split_layer.deleteFeature(actual_neighborIDs[0])
                split_layer.deleteFeature(feat_to_remove.id())
                split_layer.commitChanges()
                if split_feats[actual_neighborIDs[0]] in new_feats:
                    new_feats.remove(split_feats[actual_neighborIDs[0]])
                return (geom, actual_neighborIDs[0], False)
            elif not small or new_feat_connectors:
                neighbor = QgsFeature(fix_layer_2.fields())
                if new_feat_connectors:
                    neighbor = new_feat_connectors[-1][0]
                geom = geom.combine(neighbor.geometry())
                split_layer.startEditing()
                split_layer.deleteFeature(neighbor.id())
                split_layer.commitChanges()
                if neighbor.attributeCount() == 0:
                    return (geom, neighbor.id(), True)
                else:
                    return (geom, "break", False)
            else:
                return (geom, "break", False)

        def manualExtend(feat, neighbor_name):
            points = []
            neighbor_points = []
            makePointList(feat.geometry(), points)
            neighbor = [n for n in fix_layer_2.getFeatures() if n["Name"] == neighbor_name]
            if neighbor:
                makePointList(neighbor[0].geometry(), neighbor_points)
                closest_p = QgsPointXY()
                closest_np = QgsPointXY()
                min_dist = inf
                for pnt1 in points:
                    for pnt2 in neighbor_points:
                        if distanceArea.measureLine(QgsPointXY(pnt1[0], pnt1[1]), QgsPointXY(pnt2[0], pnt2[1])) < min_dist:
                            min_dist = distanceArea.measureLine(QgsPointXY(pnt1[0], pnt1[1]), QgsPointXY(pnt2[0], pnt2[1]))
                            closest_p = QgsPointXY(pnt1[1], pnt1[0])
                            closest_np = QgsPointXY(pnt2[1], pnt2[0])
                new_feat = QgsFeature(fix_layer_2.fields())
                new_feat.setAttributes([9999999, 'Manual Connection', 0.0, 0, 'Centerline', 0, 'Inland', NULL, NULL, NULL, NULL])
                geom = QgsGeometry().fromPolylineXY([closest_p, closest_np])
                new_feat.setGeometry(geom)
                if not closest_np.isEmpty() and not closest_p.isEmpty() and geom not in [f.geometry() for f in fix_layer_2.getFeatures(QgsFeatureRequest().setFilterExpression("Name like 'Manual Connection%'"))]:
                    fix_layer_2.startEditing()
                    fix_layer_2.dataProvider().addFeatures([new_feat])
                    fix_layer_2.commitChanges()

        def uniqueFieldAdd(layer, name, type):
            if (name not in [f.name() for f in layer.fields()]):
                layer.dataProvider().addAttributes([QgsField(name, type)])
                layer.updateFields()
        progressUpdate(3)

        #
        # -------------------------- STEP 1 -------------------------
        #



        startTime = time.time()
        # 1a) Get latest NCF from web
        fix_param_1 = {
            'INPUT': parameters['CHANNELREACH_INPUT'],
            'METHOD': 1,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        fix_result_1 = processing.run("native:fixgeometries", fix_param_1, is_child_algorithm=True, context=context)
        fix_layer_1 = context.getMapLayer(fix_result_1['OUTPUT'])
        t1 = time.time()
        progressUpdate(6, f'- Retrieved latest NCF framework from web {t1 - startTime}')
        if feedback.isCanceled():
            return {}

        fix_param_2 = {
            'INPUT': parameters['INPUT'],
            'METHOD': 1,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        fix_result_2 = processing.run("native:fixgeometries", fix_param_2, is_child_algorithm=True, context=context)
        fix_layer_2 = context.getMapLayer(fix_result_2['OUTPUT'])
        t1 = time.time()
        progressUpdate(9, f" {t1 - startTime}")
        if feedback.isCanceled():
            return {}
        
        distanceArea = QgsDistanceArea()
        project.setDistanceUnits(QgsUnitTypes.DistanceMiles)
        flagged = []
        if not bypass_split:

            # 1b) Polygons to Lines
            polygon_to_line_param = {
                'INPUT': fix_layer_1,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            polygon_result = processing.run("native:polygonstolines", polygon_to_line_param, is_child_algorithm=True, context=context)
            polygon_layer = context.getMapLayer(polygon_result['OUTPUT'])
            project.addMapLayer(polygon_layer, False)
            t1 = time.time()
            progressUpdate(12, f'- Converted Polygons to Lines {t1 - startTime}')
            if feedback.isCanceled():
                return {} 

            # 1c) Split Lines with Lines

            # Thinking through all cases as a result of splitting NCF with wtwy_lines...
            # CASE 1: Best case. 0 splits.
            # CASE 2: 1 split. Both results are big.
            # CASE 3: 1 split. One result is small. Small rejoins with big.
            # CASE 4: 1 split. Both are small. Small rejoins with other small to become big.
            # CASE 5: Multiple splits. All results are big.
            # CASE 6: Multiple splits. One result is small. Other result(s) are big. 
            #            Small rejoins with nearest big.
            # CASE 7: Multiple splits. Multiple results are small. Other result(s) are big. 
            #            7a - Small rejoins with nearest small to become big, 
            #            7b - small rejoins with nearest big,
            #            7c - small rejoins with nearest small as many times necessary to become big,
            #            7d - small rejoins with conglomeration.
            # CASE 8: Multiple splits. All results are small. Same as CASE 7c.
            # CASE 9: Worst case. 1/multiple split(s). Unable to rejoin with enough smalls 
            #            to become big. Must call in other lines_feat neighbors as many 
            #            times as necessary to become big. This case does not exist in wtwy_lines
            #            currently, but it theoretically may happen in the future as NCF changes.


            fix_layer_2.startEditing()
            to_delete = dict()
            for lines_feat in fix_layer_2.getFeatures():
                new_feats = set()
                fix_layer_2.select(lines_feat.id())

                difference_params = {
                    'INPUT': QgsProcessingFeatureSourceDefinition(fix_layer_2.id(), selectedFeaturesOnly = True, featureLimit = -1, geometryCheck = QgsFeatureRequest.GeometryAbortOnInvalid),
                    'OVERLAY': polygon_layer,
                    'OUTPUT': 'TEMPORARY_OUTPUT',
                    'GRID_SIZE': None
                }
                diff_result = processing.run("native:difference", difference_params, is_child_algorithm=True, context=context)
                diff_layer = context.getMapLayer(diff_result['OUTPUT'])
                
                split_params = {
                    'INPUT': diff_layer,
                    'LINES': polygon_layer,
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                }
                split_result = processing.run("native:splitwithlines", split_params, is_child_algorithm=True, context=context)
                split_layer = context.getMapLayer(split_result['OUTPUT'])
                fix_layer_2.removeSelection()

                split_IDs = [s.id() for s in split_layer.getFeatures()]

                # CASE 1
                if len(split_IDs) < 2:
                    continue

                split_feats = { id : split_layer.getFeature(id) for id in split_IDs }
                split_lengths = { id : distanceArea.convertLengthMeasurement(split_layer.getFeature(id).geometry().length(), QgsUnitTypes.DistanceMiles) for id in split_IDs }

                split_big_IDs = []
                split_small_IDs = []
                for key in split_lengths:
                    if split_lengths[key] > break_length:
                        split_big_IDs.append(key)
                    else:
                        split_small_IDs.append(key)

                # CASES 2, 5
                if not split_small_IDs:
                    for key in split_big_IDs:
                        new_feats.add(split_feats[key])

                # CASE 4
                elif len(split_IDs) == 2:
                    new_feat = QgsFeature(fix_layer_2.fields())
                    new_feat.setAttributes(lines_feat.attributes())
                    g, _, _ = unionSplits(next(iter(split_feats.values())).geometry(), next(iter(split_feats.values())), split_layer, split_feats, new_feats)
                    new_feat.setGeometry(g)
                    new_feats.add(new_feat)

                # CASES 3, 6, 7, 8
                else:
                    copy_small_IDs = split_small_IDs
                    for key in split_small_IDs:
                        if key in copy_small_IDs:
                            new_feat = QgsFeature(fix_layer_2.fields())
                            new_feat.setAttributes(lines_feat.attributes())
                            s = split_feats[key]
                            g = s.geometry()
                            continue_looping = True
                            while continue_looping and distanceArea.convertLengthMeasurement(g.length(), QgsUnitTypes.DistanceMiles) <= break_length: 

                                # Create spatial index each iteration. Costly, but necessary.
                                sp_idx = QgsSpatialIndex(split_layer.getFeatures())
                                neighborIDs = sp_idx.nearestNeighbor(g)
                                if key in neighborIDs:
                                    neighborIDs.remove(key)
                                
                                if neighborIDs:
                                    n = QgsFeature()
                                    found_big = False
                                    # Target big neighbor.
                                    for nID in neighborIDs:
                                        if nID in split_big_IDs:
                                            n = split_layer.getFeature(nID)
                                            found_big = True
                                            break
                                    # If no big neighbors nearby, target small neighbor.
                                    if not found_big:
                                        for nID in neighborIDs:
                                            if nID in copy_small_IDs:
                                                n = split_layer.getFeature(nID)
                                                break
                                    # Extend geometry with chosen neighbor.
                                    g = g.combine(n.geometry())
                                    # Delete chosen neighbor. From split_layer & split_feats & split_small_IDs/split_big_IDs.
                                    split_layer.startEditing()
                                    split_layer.deleteFeature(n.id())
                                    split_layer.commitChanges()
                                    if n.id() in split_feats:
                                        del split_feats[n.id()]
                                    if n.id() in copy_small_IDs:
                                        copy_small_IDs.remove(n.id())
                                    if n.id() in split_big_IDs:
                                        split_big_IDs.remove(n.id())

                                elif any(g.touches(f.geometry()) for f in new_feats): 
                                    connectors = [f for f in new_feats if g.touches(f.geometry())]
                                    g = g.combine(connectors[0].geometry())
                                    new_feats.remove(connectors[0])
                                
                                else:
                                    continue_looping = False
                                    
                                if feedback.isCanceled():
                                    return {}

                            new_feat.setGeometry(g)
                            new_feats.add(new_feat)

                    for key in split_big_IDs:
                        new_feats.add(split_feats[key])
                    if feedback.isCanceled():
                        return {}

                total_length = 0
                for f in new_feats:
                    total_length += distanceArea.convertLengthMeasurement(f.geometry().length(), QgsUnitTypes.DistanceMiles)
                if total_length > break_length:
                    to_delete[lines_feat.id()] = list(new_feats)
                
                progressUpdate(15)
                if feedback.isCanceled():
                    return {}
                
            for key in to_delete:
                fix_layer_2.addFeatures(to_delete[key])
                fix_layer_2.deleteFeature(key)
            fix_layer_2.commitChanges()

            for manual_fix in fix_layer_2.getFeatures(QgsFeatureRequest().setFilterExpression('Name = \'Curtis Bay Channel\' or Name = \'Canada - Georgian Bay 1\' or Name = \'Canada - Ontario - Parry Sound\'')):
                if manual_fix["Name"] == "Canada - Georgian Bay 1":
                    manualExtend(manual_fix, "Great Lakes - Huron 6")
                elif manual_fix["Name"] == "Canada - Ontario - Parry Sound":
                    manualExtend(manual_fix, "Canada - Georgian Bay 2")
                else:
                    manualExtend(manual_fix, "Baltimore Harbor, MD 15")
            t1 = time.time()
            progressUpdate(18, f"Split Waterway Lines Based On ChannelReach {t1 - startTime}")
            if feedback.isCanceled():
                return {} 

        # Cut off overlapping geometries.
        fix_layer_2.startEditing()
        for to_shorten in fix_layer_2.getFeatures(QgsFeatureRequest().setFilterExpression("not is_empty($geometry)").setFilterExpression("overlay_intersects(@layer)")):
            for overlap in fix_layer_2.getFeatures(QgsFeatureRequest().setFilterRect(to_shorten.geometry().boundingBox()).setFilterExpression("overlay_intersects(@layer)")):
                if to_shorten.id() != overlap.id():
                    g = to_shorten.geometry().intersection(overlap.geometry())
                    if g.length() > DEFAULT_SEGMENT_EPSILON:
                        to_shorten.setGeometry(to_shorten.geometry().difference(g))
                        fix_layer_2.updateFeature(to_shorten)
        fix_layer_2.commitChanges()
        t1 = time.time()
        progressUpdate(21, f"Cut off overlapping geometries {t1 - startTime}")
        if feedback.isCanceled():
            return {} 



        #
        # -------------------------- STEP 2 -------------------------
        #

        # 2a) Make sure distance is set to Miles in Project Properties
        project.setDistanceUnits(QgsUnitTypes.DistanceMiles)
        lines_layer = fix_layer_2.clone()
        
        # 2b) Open attributes table, abacus icon, update existing attribute, LenMiles, Geometry, re-calculate $length
        lines_layer.startEditing()
        for feature in lines_layer.getFeatures():
            feature["LenMiles"] = round(distanceArea.convertLengthMeasurement(feature.geometry().length(), QgsUnitTypes.DistanceMiles), 4)
            lines_layer.updateFeature(feature)
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(24, f"Recalculated length {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 2c) Correct linkType typos. Acceptable: (see list); flag Nulls - save flagged entries to a file and print to cmd.
        linkTypes = ["CPT", "Centerline", "Coastal-connect", "Inland", "Great Lakes/St", "International", "Internat River"]
        linkType_str = "'" + "', '".join(linkTypes) + "'"
        expr = f"(LinkType not in ({ linkType_str }) or LinkType is NULL) AND NOT Name ILIKE '%Manual Connection%'"
        lines_layer.startEditing()
        for feature in lines_layer.getFeatures(QgsFeatureRequest().setFilterExpression(expr)):
            linkType = str(feature["LinkType"])
            if feature["LinkType"] == None:
                flagged.append((feature["Name"], "Null LinkType"))
            elif 'lock' not in linkType.lower():
                matches = get_close_matches(linkType, linkTypes, n=1, cutoff=0.5)
                if matches:
                    feature["LinkType"] = matches[0]
                    lines_layer.updateFeature(feature)
                else:
                    flagged.append((feature["Name"], "Invalid LinkType"))
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(27, f"Checked for LinkType typos {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 2d) Toolbox: "Create spatial index"
        lines_layer = CreateSpatialIndex(lines_layer)
        t1 = time.time()
        progressUpdate(28, f"Created spatial index {t1 - startTime}")
        if feedback.isCanceled():
            return {} 



        #
        # -------------------------- STEP 3 -------------------------
        #

        # 3b) Remove empty and very short geometries (length <= 0.02mi)
        lines_layer.startEditing()
        for feature in lines_layer.getFeatures():
            if (feature.geometry().isNull() or feature.geometry().isEmpty() or not feature.hasGeometry() or distanceArea.convertLengthMeasurement(feature.geometry().length(), QgsUnitTypes.DistanceMiles) <= min_geom_length) and not str(feature["Name"]).startswith('Manual Connection'):
                lines_layer.deleteFeature(feature.id())
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(30, f"Removed empty & very short geometries {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 3c) Toolbox: Snap Geometries (End points only, tolerance = 0.0002 degrees)
        null_geom_params = {
            'INPUT': lines_layer,
            'REMOVE_EMPTY': False,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        null_geom_result = processing.run("native:removenullgeometries", null_geom_params, is_child_algorithm=True, context=context)
        null_geom_layer = context.getMapLayer(null_geom_result['OUTPUT'])
        fix_param_4 = {
            'INPUT': null_geom_layer,
            'METHOD': 0,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        fix_result_4 = processing.run("native:fixgeometries", fix_param_4, is_child_algorithm=True, context=context)
        fix_layer_4 = context.getMapLayer(fix_result_4['OUTPUT'])
        snap_geometries_params = {
            'INPUT': fix_layer_4,
            'REFERENCE_LAYER': fix_layer_4,
            'TOLERANCE': snapping_tolerance,
            'BEHAVIOR': 6,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        snap_geometries_layer = processing.run("native:snapgeometries", snap_geometries_params, is_child_algorithm=True, context=context)
        snap_geometries_layer = context.getMapLayer(snap_geometries_layer['OUTPUT'])
        multi_to_single_params = {
            'INPUT': snap_geometries_layer,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        multi_to_single_results = processing.run("native:multiparttosingleparts", multi_to_single_params, is_child_algorithm=True, context=context)
        multi_to_single_layer = context.getMapLayer(multi_to_single_results['OUTPUT'])
        t1 = time.time()
        progressUpdate(33, f"Snapped geometries {t1 - startTime}")
        if feedback.isCanceled():
            return {}

        # 3d) Run Disconnected Islands
        disconnectedIslandsPlugin = DisconnectedIslands(multi_to_single_layer)
        islands = []
        (disconnected_layer, islands) = disconnectedIslandsPlugin.run(disconnected_islands_tolerance)

        tolerance = snapping_tolerance
        while islands:
            mainland_layer = disconnected_layer.materialize(QgsFeatureRequest().setFilterExpression("networkGrp = 0"))
            mainland_spIdx = QgsSpatialIndex(mainland_layer.getFeatures(), flags=QgsSpatialIndex.FlagStoreFeatureGeometries)
            disconnected_layer.startEditing()
            for island in islands:
                min_dist = inf
                expr = "networkGrp = " + str(island)
                island_layer = disconnected_layer.getFeatures(QgsFeatureRequest().setFilterExpression(expr))
                for island_feat in island_layer:
                    main_neighbor = mainland_spIdx.nearestNeighbor(island_feat.geometry(), 1, tolerance)
                    if main_neighbor:
                        mainland_feat = mainland_layer.getFeature(main_neighbor[0])
                        for p1 in [QgsPointXY(v) for v in island_feat.geometry().vertices()]:
                            for p2 in [QgsPointXY(v) for v in mainland_feat.geometry().vertices()]:
                                if QgsDistanceArea().measureLine(p1, p2) < min_dist:
                                    min_dist = QgsDistanceArea().measureLine(p1, p2)
                                    closest_island_feat = island_feat
                                    closest_islandP = p1
                                    closest_mainP = p2
                if min_dist != inf:
                    closest_island_feat.setGeometry(closest_island_feat.geometry().combine(QgsGeometry().fromPolylineXY([closest_islandP, closest_mainP])))
                    disconnected_layer.updateFeature(closest_island_feat)
            disconnected_layer.commitChanges()

            multi_to_single_params = {
                'INPUT': disconnected_layer,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            multi_to_single_results = processing.run("native:multiparttosingleparts", multi_to_single_params, is_child_algorithm=True, context=context)
            multi_to_single_layer = context.getMapLayer(multi_to_single_results['OUTPUT'])
            
            disconnectedIslandsPlugin = DisconnectedIslands(multi_to_single_layer)
            islands = []
            (disconnected_layer, islands) = disconnectedIslandsPlugin.run(disconnected_islands_tolerance)
            tolerance += snapping_tolerance
            if feedback.isCanceled():
                return {}
        
        t1 = time.time()
        progressUpdate(36, f"Reconnected disconnected islands {t1 - startTime}")
        if feedback.isCanceled():
            return {} 



        #
        # -------------------------- STEP 4 -------------------------
        #
        

        # 4a) Join Channel_layer by location, 1-1 join type, maximum overlap
        fix_param_3 = {
            'INPUT': disconnected_layer,
            'METHOD': 1,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        fix_result_3 = processing.run("native:fixgeometries", fix_param_3, is_child_algorithm=True, context=context)
        fix_layer_3 = context.getMapLayer(fix_result_3['OUTPUT'])
        t1 = time.time()
        progressUpdate(42, f" {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        uniqueFieldAdd(fix_layer_3, "ChannelRea", QVariant.String)
        fix_layer_3.startEditing()
        for ncf_feat in fix_layer_1.getFeatures():
            for lines_feat  in fix_layer_3.getFeatures(QgsFeatureRequest().setFilterRect(ncf_feat.geometry().boundingBox())):
                lines_feat.setAttribute("ChannelRea", ncf_feat["channelreachidpk"])
                lines_feat.setAttribute("DepthFt", min(ncf_feat["depthmaintained"], 99))
                fix_layer_3.updateFeature(lines_feat)
        fix_layer_3.commitChanges()
        t1 = time.time()
        progressUpdate(45, f"Joined eHydro attributes by location {t1 - startTime}")
        if feedback.isCanceled():
            return {} 



        #
        # -------------------------- STEP 5 -------------------------
        #

        # 5a) For links with NCF data, depth = NCF depthmaintained
        lines_layer = fix_layer_3.clone()
        
        # 5b) For all international links, depth = 99
        lines_layer.startEditing()
        for feat in lines_layer.getFeatures(QgsFeatureRequest().setFilterExpression("LinkType = 'International' or LinkType = 'Internat River'")):
            feat["DepthFt"] = 99
            lines_layer.updateFeature(feat)
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(57, f"DepthFt attributes assigned {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 3a) Check for duplicate LinkIDs & names
        unassignableIDs = [i for i in QgsVectorLayerUtils.getValues(lines_layer, "LinkId")[0] if search(r'9{5,}', str(i)) == None]
        nextID = max(unassignableIDs) + 1
        duplicateIDs = [item for item, count in Counter(QgsVectorLayerUtils.getValues(lines_layer, "LinkId")[0]).items() if count > 1]
        duplicateNames = [item for item, count in Counter(QgsVectorLayerUtils.getValues(lines_layer, "Name")[0]).items() if count > 1]

        lines_layer.startEditing()
        for feat in lines_layer.getFeatures():
            if feat["LinkId"] not in unassignableIDs or feat["LinkId"] in duplicateIDs:
                feat["LinkId"] = nextID
                lines_layer.updateFeature(feat)
                nextID += 1
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(60, f"Checked for duplicate LinkIDs {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        lines_layer.startEditing()
        for name in duplicateNames:
            expr = f"Name = '{name}'"
            for idx, feat in enumerate(lines_layer.getFeatures(QgsFeatureRequest().setFilterExpression(expr))):
                if idx != 0:
                    feat["Name"] = name + ' ' + str(idx + 1)
                    lines_layer.updateFeature(feat)
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(63, f" {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        lines_layer.startEditing()
        for idx, feat in enumerate(lines_layer.getFeatures(QgsFeatureRequest().setFilterExpression("Name IS NULL"))):
            feat["Name"] = str(idx)
            lines_layer.updateFeature(feat)
        lines_layer.commitChanges()
        t1 = time.time()
        progressUpdate(66, f"Checked for duplicate names {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # Remove Plot, Domestic, Deepdraft attributes
        if lines_layer.fields().indexFromName('Plot') != -1:
            lines_layer.dataProvider().deleteAttributes([lines_layer.fields().indexFromName('Plot')])
            lines_layer.updateFields()
        if lines_layer.fields().indexFromName('Domestic') != -1:
            lines_layer.dataProvider().deleteAttributes([lines_layer.fields().indexFromName('Domestic')])
            lines_layer.updateFields()
        if lines_layer.fields().indexFromName('Deepdraft') != -1:
            lines_layer.dataProvider().deleteAttributes([lines_layer.fields().indexFromName('Deepdraft')])
            lines_layer.updateFields()
        t1 = time.time()
        progressUpdate(78, f"Removed Plot, Domestic, Deepdraft attributes {t1 - startTime}")
        if feedback.isCanceled():
            return {} 



        #
        # -------------------------- STEP 6 -------------------------
        #

        date_str = date.today().strftime('%Y%m%d')
        t1 = time.time()
        progressUpdate(84, f" {t1 - startTime}")
        if feedback.isCanceled():
            return {} 
        
        # 6a) Use QGIS Networks plugin (Build Graph)
        nodes_params_1 = {
            'RESEAU': lines_layer,
            'SENS': '',
            'IDENT': 0,
            'PREFIXE': '',
            'DECIMALES': 9
        }
        networkPlugin = Networks()
        (lines_layer, nodes_layer) = networkPlugin.build_graph(nodes_params_1)
        t1 = time.time()
        progressUpdate(87, f"Built Graph with Networks plugin {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 5c) All others: deepest depth of surrounding neighbors
        def maxNeighborsDepth(layer):
            layer.startEditing()
            for nullDepth_feat in layer.getFeatures(QgsFeatureRequest().setFilterExpression("DepthFt IS NULL")):
                depths = []
                for hasDepth_neighbor in layer.getFeatures(QgsFeatureRequest().setFilterExpression("DepthFt IS NOT NULL").setFilterRect(nullDepth_feat.geometry().boundingBox())):
                    depths.append(hasDepth_neighbor["DepthFt"])
                if depths:
                    nullDepth_feat["DepthFt"] = max(depths)
                    layer.updateFeature(nullDepth_feat)
            layer.commitChanges()
        
        while list(lines_layer.getFeatures(QgsFeatureRequest().setFilterExpression("DepthFt IS NULL"))):
            maxNeighborsDepth(lines_layer)
            if feedback.isCanceled():
                return {}
            
        t1 = time.time()
        progressUpdate(90, f"Null DepthFt attributes estimated {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 6b) Nodes: Add lat-lon attributes
        nodes_layer.dataProvider().addAttributes([QgsField("Latitude", QVariant.Double), QgsField("Longitude", QVariant.Double)])
        nodes_layer.updateFields()
        nodes_layer.startEditing()
        for feature in nodes_layer.getFeatures():
            point = QgsPointXY(QgsGeometry.asPoint(feature.geometry()))
            feature["Longitude"] = float('%.6f'%(point.x()))
            feature["Latitude"] = float('%.6f'%(point.y()))
            nodes_layer.updateFeature(feature)
        nodes_layer.commitChanges()
        t1 = time.time()
        progressUpdate(93, f"Lat-lon attributes created & assigned to nodes {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 6c) Create spatial index
        lines_layer = CreateSpatialIndex(lines_layer)
        t1 = time.time()
        progressUpdate(96, f"Created spatial index {t1 - startTime}")
        if feedback.isCanceled():
            return {} 

        # 6d) Nodes: Add "State abbreviation/FIPS", "County Name" attributes       
        if state_exists and state_abbr != '' and state_fips != '':
            nodes_layer.dataProvider().addAttributes([QgsField("StateAbbr", QVariant.String), QgsField("StateFIPS", QVariant.Int)])
            nodes_layer.updateFields()
            state_params = {
                'INPUT': nodes_layer,
                'PREDICATE': [0],
                'JOIN': state_layer,
                'JOIN_FIELDS': [],
                'METHOD': 2,
                'DISCARD_NONMATCHING': False,
                'PREFIX': '',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            state_result = processing.run("native:joinattributesbylocation", state_params, is_child_algorithm=True, context=context)
            state_layer = context.getMapLayer(state_result['OUTPUT'])

            nodes_layer.startEditing()
            for state_feat in state_layer.getFeatures():
                if state_feat[state_abbr] != None and state_feat[state_fips] != None:
                    for nodes_feat in nodes_layer.getFeatures(QgsFeatureRequest().setFilterRect(state_feat.geometry().boundingBox())):
                        nodes_feat.setAttribute("StateAbbr", state_feat[state_abbr])
                        nodes_feat.setAttribute("StateFIPS", state_feat[state_fips])
                        nodes_layer.updateFeature(nodes_feat)
            nodes_layer.commitChanges()
            t1 = time.time()
            progressUpdate(97, f"State Abbreviation/FIPS attributes created & assigned to nodes {t1 - startTime}")
            if feedback.isCanceled():
                return {} 

        if county_exists and county_name != '' and county_fips != '':
            nodes_layer.dataProvider().addAttributes([QgsField("CountyName", QVariant.String), QgsField("CountyFIPS", QVariant.Int)])
            nodes_layer.updateFields()
            county_params = {
                'INPUT': nodes_layer,
                'PREDICATE': [0],
                'JOIN': county_layer,
                'JOIN_FIELDS': [],
                'METHOD': 2,
                'DISCARD_NONMATCHING': False,
                'PREFIX': '',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            county_result = processing.run("native:joinattributesbylocation", county_params, is_child_algorithm=True, context=context)
            county_layer = context.getMapLayer(county_result['OUTPUT'])

            nodes_layer.startEditing()
            for county_feat in county_layer.getFeatures():
                if county_feat[county_name] != None and county_feat[county_fips] != None:
                    for nodes_feat in nodes_layer.getFeatures(QgsFeatureRequest().setFilterRect(county_feat.geometry().boundingBox())):
                        nodes_feat.setAttribute("CountyName", county_feat[county_name])
                        nodes_feat.setAttribute("CountyFIPS", county_feat[county_fips])
                        nodes_layer.updateFeature(nodes_feat)
            nodes_layer.commitChanges()
            t1 = time.time()
            progressUpdate(98, f"County Name attributes created & assigned to nodes {t1 - startTime}")
            if feedback.isCanceled():
                return {}

        if country_exists and country_abbr != '':
            nodes_layer.dataProvider().addAttributes([QgsField("CountryAbbr", QVariant.String)])
            nodes_layer.updateFields()
            country_params = {
                'INPUT': nodes_layer,
                'PREDICATE': [0],
                'JOIN': country_layer,
                'JOIN_FIELDS': [],
                'METHOD': 2,
                'DISCARD_NONMATCHING': False,
                'PREFIX': '',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            country_result = processing.run("native:joinattributesbylocation", country_params, is_child_algorithm=True, context=context)
            country_layer = context.getMapLayer(country_result['OUTPUT'])

            nodes_layer.startEditing()
            for country_feat in country_layer.getFeatures():
                if country_feat[country_abbr] != None:
                    for nodes_feat in nodes_layer.getFeatures(QgsFeatureRequest().setFilterRect(country_feat.geometry().boundingBox())):
                        nodes_feat.setAttribute("CountryAbbr", country_feat[country_abbr])
                        nodes_layer.updateFeature(nodes_feat)
            nodes_layer.commitChanges()
            t1 = time.time()
            progressUpdate(99, f"Country Abbreviation/FIPS attributes created & assigned to nodes {t1 - startTime}")
            if feedback.isCanceled():
                return {}



        #
        # -------------------------- END ----------------------------
        #

        # Remove networkGrp attributes
        if lines_layer.fields().indexFromName('networkGrp') != -1:
            lines_layer.dataProvider().deleteAttributes([lines_layer.fields().indexFromName('networkGrp')])
            lines_layer.updateFields()
        if lines_layer.fields().indexFromName('fid') != -1:
            lines_layer.dataProvider().deleteAttributes([lines_layer.fields().indexFromName('fid')])
            lines_layer.updateFields()
        if nodes_layer.fields().indexFromName('fid') != -1:
            nodes_layer.dataProvider().deleteAttributes([nodes_layer.fields().indexFromName('fid')])
            nodes_layer.updateFields()
            
        # 2b) Open attributes table, abacus icon, update existing attribute, LenMiles, Geometry, re-calculate $length
        lines_layer.startEditing()
        for feature in lines_layer.getFeatures():
            feature["LenMiles"] = round(distanceArea.convertLengthMeasurement(feature.geometry().length(), QgsUnitTypes.DistanceMiles), 4)
            lines_layer.updateFeature(feature)
        lines_layer.commitChanges()
        
        refactor_parameters_1 = {
            'INPUT': lines_layer,
            'FIELDS_MAPPING': [
                {'alias': '','comment': '','expression': '"LinkId"','length': 0,'name': 'LinkId','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'},
                {'alias': '','comment': '','expression': '"Name"','length': 254,'name': 'Name','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},
                {'alias': '','comment': '','expression': '"LenMiles"','length': 20,'name': 'LenMiles','precision': 4,'sub_type': 0,'type': 6,'type_name': 'double precision'},
                {'alias': '','comment': '','expression': '"DepthFt"','length': 20,'name': 'DepthFt','precision': 4,'sub_type': 0,'type': 6,'type_name': 'double precision'},
                {'alias': '','comment': '','expression': '"LinkType"','length': 80,'name': 'LinkType','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},
                {'alias': '','comment': '','expression': '"i"','length': 80,'name': 'i','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},
                {'alias': '','comment': '','expression': '"j"','length': 80,'name': 'j','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},
                {'alias': '','comment': '','expression': '"ij"','length': 80,'name': 'ij','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'},
                {'alias': '','comment': '','expression': '"ChannelRea"','length': 25,'name': 'ChannelRea','precision': 0,'sub_type': 0,'type': 10,'type_name': 'text'}
            ],
            'OUTPUT':'TEMPORARY_OUTPUT'
        }
        refactor_results = processing.run("native:refactorfields", refactor_parameters_1, is_child_algorithm=True, context=context)
        lines_layer = context.getMapLayer(refactor_results['OUTPUT'])
        
        (link_sink, link_dest_id) = self.parameterAsSink(parameters, 'OUTPUT', context, lines_layer.fields(), lines_layer.wkbType(), lines_layer.sourceCrs())
        (node_sink, node_dest_id) = self.parameterAsSink(parameters, 'NODE_OUTPUT', context, nodes_layer.fields(), nodes_layer.wkbType(), nodes_layer.sourceCrs())
        
        #https://gis.stackexchange.com/questions/415841/changing-output-layers-name-in-qgis-processing-plugin
        link_layer_details = context.layerToLoadOnCompletionDetails(link_dest_id)
        link_layer_details.name = f"{date_str}_network_links"
        node_layer_details = context.layerToLoadOnCompletionDetails(node_dest_id)
        node_layer_details.name = f"{date_str}_network_nodes"

        for feat in lines_layer.getFeatures():
            link_sink.addFeature(feat, QgsFeatureSink.FastInsert)
        for feat in nodes_layer.getFeatures():
            node_sink.addFeature(feat, QgsFeatureSink.FastInsert)
        t1 = time.time()
        progressUpdate(100, f"Removed networkGrp attributes {t1 - startTime}")
        if feedback.isCanceled():
            return {}
                
        if flagged:
            feedback.setProgressText("The following Links should be manually reviewed for the following reasons:")
            for x in flagged:
                feedback.setProgressText(f"{x[0]}: {x[1]}")

        return {
            'OUTPUT': link_dest_id,
            'NODE_OUTPUT': node_dest_id,
            'FLAGGED': ", ".join(str(x[0]) + ": " + str(x[1]) for x in flagged)
        }