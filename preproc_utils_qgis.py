from PyQt5.QtCore import *
from PyQt5.QtCore import QVariant
from qgis.core import QgsField


def get_layer_by_name(layer_name):
    layer = None
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.name() == layer_name:
            layer = lyr
            break
    return layer


def set_group_sequential_number(layer, id_col, group_col, group_sid_col = "GR_SID", offset=1):
    features = list(layer.getFeatures())
    # obtain unique groups
    groups = []
    for feat in features:
        id  = feat[id_col]
        group = feat[group_col]
        if (id != NULL) and (group != NULL):
            groups.append(group)

    groups = list(set(groups))
    groups.sort()
    # set sequential number starting in 1
    groups_to_sid = {group: i+offset for i, group in enumerate(groups)}

    # edit file
    index_attribute = layer.fields().indexFromName(group_sid_col)
    if index_attribute < 0:
        layer.startEditing()
        group_sid_field = QgsField(group_sid_col, QVariant.Int)
        layer.dataProvider().addAttributes([group_sid_field])
        layer.updateFields()
        layer.commitChanges()
        index_attribute = layer.fields().indexFromName(group_sid_col)
    print("col: {}, index_attribute {}".format(group_sid_col, index_attribute))

    layer.startEditing()
    for feat in features:
        id = feat[id_col]
        group = feat[group_col]
        if (id != NULL) and (group != NULL):
            group_sid = groups_to_sid[group]
            layer.changeAttributeValue(feat.id(), index_attribute, group_sid)
    layer.commitChanges()


def set_sequential_number(layer, id_col, sid_col, offset=0):
    features = list(layer.getFeatures())
    seq_number = offset
    index_attribute = layer.fields().indexFromName(sid_col)

    ids = []
    for feat in features:
        id = feat[id_col]
        if id != NULL:
            ids.append(id)
    ids.sort()
    id_to_sid = {id: i + offset for i, id in enumerate(ids)}
    print(len(ids))

    layer.startEditing()
    for feat in features:
        id = feat[id_col]
        if id != NULL:
            layer.changeAttributeValue(feat.id(), index_attribute, id_to_sid[id])
            seq_number += 1
    print(seq_number)
    layer.commitChanges()


def add_int_attributes(layer, attribute_names):
    pr = layer.dataProvider()
    layer.startEditing()
    attribute_list = []
    for att_name in attribute_names:
        index_attribute = layer.fields().indexFromName(att_name)
        if index_attribute < 0:
            attribute_list.append(QgsField(att_name, QVariant.Int))
    pr.addAttributes(attribute_list)
    layer.commitChanges()


def add_sequential_and_group_id_TZA():
    # shapefile_path = "~/data/wpop/OtherBoundaries/TZA/tza_admbnda_adm3_20181019/tza_admbnda_adm3_20181019.shp"
    layer = get_layer_by_name("tza_admbnda_adm3_20181019")
    add_int_attributes(layer, ["SID", "GR_SID"])
    set_sequential_number(layer, "ADM3_PCODE", "SID", offset=1)
    set_group_sequential_number(layer, "ADM3_PCODE", "ADM2_PCODE", group_sid_col="GR_SID", offset=1)


def add_sequential_and_group_id_COD():
    # shapefile_path = "~/data/wpop/OtherBoundaries/COD/cod_adm2/cod_admbnda_adm2_rgc_20190911.shp"
    layer = get_layer_by_name("cod_admbnda_adm2_rgc_20190911")
    add_int_attributes(layer, ["SID", "GR_SID"])
    set_sequential_number(layer, "ADM2_PCODE", "SID", offset=1)
    set_group_sequential_number(layer, "ADM2_PCODE", "ADM1_PCODE", group_sid_col="GR_SID", offset=1)


def add_sequential_and_group_RWA():
    # shapefile_path = "~/data/wpop/OtherBoundaries/RWA/rwa_adm_2006_nisr_wgs1984_20181002_shp/rwa_adm3_2006_NISR_WGS1984_20181002.shp"
    layer = get_layer_by_name("rwa_adm3_2006_NISR_WGS1984_20181002")
    add_int_attributes(layer, ["SID", "GR_SID"])
    set_sequential_number(layer, "ADM3_PCODE", "SID", offset=1)
    set_group_sequential_number(layer, "ADM3_PCODE", "ADM2_PCODE", group_sid_col="GR_SID", offset=1)


def add_sequential_and_group_MOZ():
    # shapefile_path = "~/data/wpop/OtherBoundaries/MOZ/moz_adm3/moz_admbnda_adm3_ine_20190607.shp"
    layer = get_layer_by_name("moz_admbnda_adm3_ine_20190607")
    add_int_attributes(layer, ["SID", "GR_SID"])
    set_sequential_number(layer, "ADM3_PCODE", "SID", offset=1)
    set_group_sequential_number(layer, "ADM3_PCODE", "ADM2_PCODE", group_sid_col="GR_SID", offset=1)


def add_sequential_and_group_NGA():
    # shapefile_path = "~/data/wpop/OtherBoundaries/NGA/nga_adm2/nga_polnda_adm2_1m_salb.shp"
    layer = get_layer_by_name("nga_polnda_adm2_1m_salb")
    add_int_attributes(layer, ["SID", "GR_SID"])
    set_sequential_number(layer, "ADM2_CODE", "SID", offset=1)
    set_group_sequential_number(layer, "ADM2_CODE", "ADM1_CODE", group_sid_col="GR_SID", offset=1)


def add_sequential_and_group_UGA():
    # shapefile_path = "~/data/wpop/OtherBoundaries/UGA/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm4_ubos_20200824.shp"
    layer = get_layer_by_name("uga_admbnda_adm4_ubos_20200824")
    add_int_attributes(layer, ["SID", "GR_SID"])
    set_sequential_number(layer, "ADM4_PCODE", "SID", offset=1)
    set_group_sequential_number(layer, "ADM4_PCODE", "ADM3_PCODE", group_sid_col="GR_SID", offset=1)

