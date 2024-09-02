# Form implementation generated from reading ui file 'ui/Lobby.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(435, 496)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(parent=Dialog)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(parent=Dialog)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(20)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.stackedWidget = QtWidgets.QStackedWidget(parent=Dialog)
        self.stackedWidget.setObjectName("stackedWidget")
        self.StartPage = QtWidgets.QWidget()
        self.StartPage.setObjectName("StartPage")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.StartPage)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(parent=self.StartPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(16)
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.newProjectButton = QtWidgets.QPushButton(parent=self.StartPage)
        self.newProjectButton.setObjectName("newProjectButton")
        self.verticalLayout_2.addWidget(self.newProjectButton)
        self.openProjectButton = QtWidgets.QPushButton(parent=self.StartPage)
        self.openProjectButton.setObjectName("openProjectButton")
        self.verticalLayout_2.addWidget(self.openProjectButton)
        self.label_4 = QtWidgets.QLabel(parent=self.StartPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(16)
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.recentProjectsList = QtWidgets.QListView(parent=self.StartPage)
        self.recentProjectsList.setAutoFillBackground(True)
        self.recentProjectsList.setObjectName("recentProjectsList")
        self.verticalLayout_2.addWidget(self.recentProjectsList)
        self.stackedWidget.addWidget(self.StartPage)
        self.NewProjectPage = QtWidgets.QWidget()
        self.NewProjectPage.setObjectName("NewProjectPage")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.NewProjectPage)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(parent=self.NewProjectPage)
        self.groupBox.setObjectName("groupBox")
        self.formLayout = QtWidgets.QFormLayout(self.groupBox)
        self.formLayout.setObjectName("formLayout")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_5)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.projectNameField = QtWidgets.QLineEdit(parent=self.groupBox)
        self.projectNameField.setObjectName("projectNameField")
        self.horizontalLayout_3.addWidget(self.projectNameField)
        self.label_7 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_3.addWidget(self.label_7)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)
        self.label_6 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.projectLocationField = QtWidgets.QLineEdit(parent=self.groupBox)
        self.projectLocationField.setMinimumSize(QtCore.QSize(0, 18))
        self.projectLocationField.setObjectName("projectLocationField")
        self.horizontalLayout_2.addWidget(self.projectLocationField)
        self.projectLocationButton = QtWidgets.QPushButton(parent=self.groupBox)
        self.projectLocationButton.setMaximumSize(QtCore.QSize(20, 20))
        self.projectLocationButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui/../icons/folder.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.projectLocationButton.setIcon(icon)
        self.projectLocationButton.setIconSize(QtCore.QSize(18, 18))
        self.projectLocationButton.setFlat(True)
        self.projectLocationButton.setObjectName("projectLocationButton")
        self.horizontalLayout_2.addWidget(self.projectLocationButton)
        self.formLayout.setLayout(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.horizontalLayout_2)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.NewProjectPage)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.passwordProtectCheckbox = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.passwordProtectCheckbox.setObjectName("passwordProtectCheckbox")
        self.verticalLayout_3.addWidget(self.passwordProtectCheckbox)
        self.widget = QtWidgets.QWidget(parent=self.groupBox_2)
        self.widget.setObjectName("widget")
        self.formLayout_3 = QtWidgets.QFormLayout(self.widget)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_8 = QtWidgets.QLabel(parent=self.widget)
        self.label_8.setObjectName("label_8")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_8)
        self.passwordField = QtWidgets.QLineEdit(parent=self.widget)
        self.passwordField.setObjectName("passwordField")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.passwordField)
        self.label_9 = QtWidgets.QLabel(parent=self.widget)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_9)
        self.confirmPasswordField = QtWidgets.QLineEdit(parent=self.widget)
        self.confirmPasswordField.setObjectName("confirmPasswordField")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.confirmPasswordField)
        self.verticalLayout_3.addWidget(self.widget)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancelButton = QtWidgets.QPushButton(parent=self.NewProjectPage)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout.addWidget(self.cancelButton)
        self.createProjectButton = QtWidgets.QPushButton(parent=self.NewProjectPage)
        self.createProjectButton.setObjectName("createProjectButton")
        self.horizontalLayout.addWidget(self.createProjectButton)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.stackedWidget.addWidget(self.NewProjectPage)
        self.verticalLayout.addWidget(self.stackedWidget)

        self.retranslateUi(Dialog)
        self.stackedWidget.setCurrentIndex(1)
        self.passwordProtectCheckbox.toggled['bool'].connect(self.widget.setVisible) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Welcome To"))
        self.label_2.setText(_translate("Dialog", "PythonCBAS"))
        self.label_3.setText(_translate("Dialog", "START"))
        self.newProjectButton.setText(_translate("Dialog", "New Project"))
        self.openProjectButton.setText(_translate("Dialog", "Open Project"))
        self.label_4.setText(_translate("Dialog", "RECENT"))
        self.groupBox.setTitle(_translate("Dialog", "New Project"))
        self.label_5.setText(_translate("Dialog", "Name:"))
        self.label_7.setText(_translate("Dialog", ".cbasproj"))
        self.label_6.setText(_translate("Dialog", "Location:"))
        self.groupBox_2.setTitle(_translate("Dialog", "Advanced"))
        self.passwordProtectCheckbox.setText(_translate("Dialog", "Password Protect"))
        self.label_8.setText(_translate("Dialog", "Password:"))
        self.label_9.setText(_translate("Dialog", "Confirm Password:"))
        self.cancelButton.setText(_translate("Dialog", "Cancel"))
        self.createProjectButton.setText(_translate("Dialog", "Create"))